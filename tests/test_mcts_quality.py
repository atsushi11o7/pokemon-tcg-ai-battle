import unittest
from types import SimpleNamespace

from training.mcts.determinize import determinize_for_search
from training.mcts.search import MAX_ACTIONS_PER_NODE, _enumerate_actions
from training.mcts.train import should_accept_candidate


def card(card_id: int, player_index: int = 0):
    return SimpleNamespace(id=card_id, playerIndex=player_index)


def pokemon(card_id: int, player_index: int = 0):
    return SimpleNamespace(
        id=card_id,
        preEvolution=[card(card_id + 1, player_index)],
        energyCards=[card(card_id + 2, player_index)],
        tools=[card(card_id + 3, player_index)],
    )


class MctsQualityTest(unittest.TestCase):
    def test_determinization_respects_real_decks_and_visible_cards(self) -> None:
        own_deck = list(range(1, 61))
        opponent_deck = list(range(101, 161))
        own = SimpleNamespace(
            active=[pokemon(2)],
            bench=[],
            deckCount=47,
            discard=[card(6)],
            prize=[None] * 6,
            handCount=1,
            hand=[card(1)],
        )
        opponent = SimpleNamespace(
            active=[SimpleNamespace(id=101, preEvolution=[], energyCards=[], tools=[])],
            bench=[],
            deckCount=52,
            discard=[],
            prize=[None] * 6,
            handCount=1,
            hand=None,
        )
        state = SimpleNamespace(players=[own, opponent], stadium=[card(7, 0)], yourIndex=0)
        obs = SimpleNamespace(current=state)

        kwargs, assumed_decks = determinize_for_search(obs, own_deck, [opponent_deck])

        self.assertEqual(len(kwargs["your_deck"]), 47)
        self.assertEqual(len(kwargs["your_prize"]), 6)
        self.assertEqual(len(kwargs["opponent_deck"]), 52)
        self.assertEqual(len(kwargs["opponent_hand"]), 1)
        self.assertEqual(len(kwargs["opponent_prize"]), 6)
        self.assertFalse({1, 2, 3, 4, 5, 6, 7} & set(kwargs["your_deck"]))
        self.assertEqual(assumed_decks, [own_deck, opponent_deck])

    def test_action_cap_samples_full_combination_rank_space(self) -> None:
        select = SimpleNamespace(minCount=2, maxCount=4, option=[object()] * 20)
        actions = _enumerate_actions(select)

        self.assertEqual(len(actions), MAX_ACTIONS_PER_NODE)
        self.assertTrue(any(0 in action for action in actions))
        self.assertTrue(any(19 in action for action in actions))
        self.assertEqual(len({tuple(action) for action in actions}), len(actions))

    def test_candidate_must_beat_current_best(self) -> None:
        self.assertFalse(should_accept_candidate(0.3, 0.8))
        self.assertFalse(should_accept_candidate(0.5, 0.8))
        self.assertTrue(should_accept_candidate(0.6, 0.6))


if __name__ == "__main__":
    unittest.main()
