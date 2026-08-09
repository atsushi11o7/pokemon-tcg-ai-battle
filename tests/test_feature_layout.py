import unittest
from pathlib import Path
from types import SimpleNamespace

from training.common import sparse_features as sf

ROOT = Path(__file__).resolve().parents[1]


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


class FeatureLayoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self._original = sf.feature_layout()

    def tearDown(self) -> None:
        sf.configure_feature_layout(self._original)

    def test_shared_layout_is_much_smaller(self) -> None:
        sf.configure_feature_layout("per_role")
        per_role = sf.encoder_size() + sf.decoder_size()
        sf.configure_feature_layout("shared_card")
        shared = sf.encoder_size() + sf.decoder_size()
        self.assertLess(shared * 4, per_role)

    def test_unknown_layout_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sf.configure_feature_layout("nonexistent")

    def test_shared_layout_keeps_card_roles_separable(self) -> None:
        """同一トークン内で同居する役割は、別ブロックに落ちること。

        本体/道具/エネルギーが同じブロックへ混ざると、EmbeddingBagのsumで
        どのカードがどの役割だったかの対応が失われる。
        """
        sf.configure_feature_layout("shared_card")
        count = sf.card_count()
        roles = (sf.CARD_ROLE_POKEMON, sf.CARD_ROLE_TOOL, sf.CARD_ROLE_ENERGY)
        blocks = {role * count // count for role in roles}
        self.assertEqual(len(blocks), len(roles))
        self.assertEqual(sf.ENCODER_CARD_BLOCKS, 4)


class DeclaredSizeMatchesWritesTest(unittest.TestCase):
    """宣言サイズと実際の書き込み量が一致すること。

    `encoder_size()`は`get_encoder_input`の書き込みと手作業で辻褄を合わせている。
    ずれるとブロック境界が重なり、無関係な特徴が同じ埋め込み行を共有して静かに壊れる。
    実盤面で`SparseVector.pos`と突き合わせて、その手作業を検証する。
    """

    def _observation(self):
        from cg.api import to_observation_class
        from cg.game import battle_finish, battle_start

        from training.common.deck import parse_deck_csv

        deck = parse_deck_csv(ROOT / "decks" / "candidates" / "crustle_meta.csv")
        obs_dict, _start = battle_start(deck, deck)
        try:
            return to_observation_class(obs_dict), deck
        finally:
            battle_finish()

    def test_encoder_size_matches_written_positions(self) -> None:
        original = sf.feature_layout()
        obs, deck = self._observation()
        try:
            for layout in ("per_role", "shared_card"):
                sf.configure_feature_layout(layout)
                written = sf.get_encoder_input(obs, deck)
                self.assertEqual(
                    written.pos,
                    sf.encoder_size(),
                    f"{layout}: encoder_size()が実際の書き込み量と一致しない",
                )
                self.assertLess(max(written.index), sf.encoder_size())
        finally:
            sf.configure_feature_layout(original)


if __name__ == "__main__":
    unittest.main()
