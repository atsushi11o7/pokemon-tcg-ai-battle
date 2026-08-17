import random
import unittest
from pathlib import Path
from types import SimpleNamespace

from training.common import sparse_features as sf

ROOT = Path(__file__).resolve().parents[1]


def _token_values(sparse_vector) -> list[list[float]]:
    """トークンごとに書き込まれた値を、添字に依らない形(ソート済み)で取り出す。"""
    boundaries = [*sparse_vector.offset, len(sparse_vector.index)]
    return [
        sorted(sparse_vector.value[start:end])
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]


class EffectiveDamageTest(unittest.TestCase):
    """弱点・抵抗力を反映した実効ダメージ(KO判定の土台)。"""

    def card(self, energy_type=None, weakness=None, resistance=None):
        return SimpleNamespace(energyType=energy_type, weakness=weakness, resistance=resistance)

    def test_weakness_doubles_damage(self) -> None:
        attacker = self.card(energy_type=1)
        defender = self.card(weakness=1)
        self.assertEqual(sf.effective_damage(60.0, attacker, defender), 120.0)

    def test_resistance_reduces_damage(self) -> None:
        attacker = self.card(energy_type=2)
        defender = self.card(resistance=2)
        self.assertEqual(sf.effective_damage(60.0, attacker, defender), 30.0)

    def test_unrelated_type_is_unchanged(self) -> None:
        attacker = self.card(energy_type=3)
        defender = self.card(weakness=1, resistance=2)
        self.assertEqual(sf.effective_damage(60.0, attacker, defender), 60.0)

    def test_resistance_never_goes_negative(self) -> None:
        attacker = self.card(energy_type=2)
        defender = self.card(resistance=2)
        self.assertEqual(sf.effective_damage(10.0, attacker, defender), 0.0)

    def test_zero_damage_attack_stays_zero(self) -> None:
        attacker = self.card(energy_type=1)
        defender = self.card(weakness=1)
        self.assertEqual(sf.effective_damage(0.0, attacker, defender), 0.0)


class HpRatioBucketTest(unittest.TestCase):
    """残りHP比の離散化。閾値的な判断のため生スカラーと併用する。"""

    def test_full_and_empty_map_to_ends(self) -> None:
        self.assertEqual(sf._hp_ratio_bucket(0.0), 0)
        self.assertEqual(sf._hp_ratio_bucket(1.0), sf.HP_RATIO_BUCKETS - 1)

    def test_bucket_is_monotonic(self) -> None:
        buckets = [sf._hp_ratio_bucket(r / 20) for r in range(21)]
        self.assertEqual(buckets, sorted(buckets))

    def test_bucket_stays_in_range(self) -> None:
        for ratio in (-1.0, 0.0, 0.5, 1.0, 2.0):
            self.assertIn(sf._hp_ratio_bucket(ratio), range(sf.HP_RATIO_BUCKETS))


class DeclaredSizeMatchesWritesTest(unittest.TestCase):
    """宣言サイズと実際の書き込み量が一致すること。

    `encoder_size()`は`get_encoder_input`の書き込みと手作業で辻褄を合わせており、
    ずれるとブロック境界が重なって無関係な特徴が同じ埋め込み行を共有し、静かに壊れる。

    `battle_start()`直後はコイントス選択で場が空なので、`add_pokemon`の非空枝も
    `effective_damage`も一度も実行されない。実際に手を進めて、場にポケモンが
    並んだ局面で検証する。
    """

    def _mid_game_observations(self, count: int = 400):
        from cg.api import to_observation_class
        from cg.game import battle_finish, battle_select, battle_start

        from training.common.deck import parse_deck_csv

        deck = parse_deck_csv(ROOT / "decks" / "candidates" / "crustle_meta.csv")
        random.seed(0)
        collected = []
        obs_dict, _start = battle_start(deck, deck)
        try:
            obs = to_observation_class(obs_dict)
            steps = 0
            while obs.current.result < 0 and len(collected) < count and steps < 400:
                state = obs.current
                if any(p.active and p.active[0] is not None for p in state.players):
                    collected.append(obs_dict)
                select = obs.select
                take = min(max(select.minCount, 1), len(select.option))
                obs_dict = battle_select(list(range(take)))
                obs = to_observation_class(obs_dict)
                steps += 1
        finally:
            battle_finish()
        return collected, deck

    def test_encoder_size_matches_written_positions(self) -> None:
        from cg.api import to_observation_class

        observations, deck = self._mid_game_observations()
        self.assertGreater(len(observations), 0, "場にポケモンが並んだ局面を取得できなかった")
        for obs_dict in observations:
            obs = to_observation_class(obs_dict)
            written = sf.get_encoder_input(obs, deck)
            self.assertEqual(
                written.pos, sf.encoder_size(), "encoder_size()が実際の書き込み量と一致しない"
            )
            self.assertLess(max(written.index), sf.encoder_size())
            self.assertGreaterEqual(min(written.index), 0)

    def test_decoder_indices_stay_in_range(self) -> None:
        from cg.api import to_observation_class

        observations, _deck = self._mid_game_observations()
        for obs_dict in observations:
            obs = to_observation_class(obs_dict)
            actions = [[i] for i in range(len(obs.select.option))] or [[]]
            written = sf.get_decoder_input(obs, actions)
            if not written.index:
                continue
            self.assertLess(max(written.index), sf.decoder_size())
            self.assertGreaterEqual(min(written.index), 0)

    def test_token_count_matches_network_definition(self) -> None:
        """書き込むトークン数と`network.py`のowner/zone定義が一致すること。

        ずれると`reshape`で無関係なトークンが混ざるか、実行時に落ちる。
        """
        from cg.api import to_observation_class

        from training.common import network

        observations, deck = self._mid_game_observations()
        for obs_dict in observations:
            written = sf.get_encoder_input(to_observation_class(obs_dict), deck)
            self.assertEqual(len(written.offset), network.NUM_WORDS_ENCODER)
        self.assertEqual(len(network._TOKEN_OWNER_ZONE), network.NUM_WORDS_ENCODER + 1)

    def test_non_empty_pokemon_branch_is_actually_exercised(self) -> None:
        """上の2テストが場の空な局面だけを見ていないことを保証する。"""
        from cg.api import to_observation_class

        observations, _deck = self._mid_game_observations()
        occupied = 0
        for obs_dict in observations:
            state = to_observation_class(obs_dict).current
            for player in state.players:
                occupied += sum(
                    1 for p in (list(player.active or []) + list(player.bench or [])) if p
                )
        self.assertGreater(occupied, 0)


if __name__ == "__main__":
    unittest.main()
