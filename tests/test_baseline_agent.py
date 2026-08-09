import unittest
from types import SimpleNamespace
from unittest.mock import patch

from evaluation.match_runner import first_index_agent_factory


def observation(min_count: int, max_count: int, n_options: int):
    return SimpleNamespace(
        select=SimpleNamespace(
            option=list(range(n_options)), minCount=min_count, maxCount=max_count
        )
    )


class FirstIndexAgentTest(unittest.TestCase):
    """常に先頭の選択肢を返す基準線。

    `vs random`は早々に100%へ飽和して学習の進捗を検出できなくなるため、
    これを物差しに使う。実測では#14の重みでもこの相手に60.8%しか勝てておらず、
    ランダム相手の96.7%では見えなかった差を捉えられる。
    """

    def agent_for(self, min_count: int, max_count: int, n_options: int):
        deck = [1] * 60
        agent = first_index_agent_factory(deck)
        obs = observation(min_count, max_count, n_options)
        with patch("evaluation.match_runner.to_observation_class", return_value=obs):
            return agent({})

    def test_single_choice_returns_first_index(self) -> None:
        self.assertEqual(self.agent_for(1, 1, 5), [0])

    def test_optional_choice_still_returns_one(self) -> None:
        self.assertEqual(self.agent_for(0, 3, 5), [0])

    def test_multi_select_satisfies_min_count(self) -> None:
        self.assertEqual(self.agent_for(3, 4, 5), [0, 1, 2])

    def test_never_exceeds_available_options(self) -> None:
        self.assertEqual(self.agent_for(5, 5, 2), [0, 1])

    def test_deck_is_returned_before_the_game_starts(self) -> None:
        deck = list(range(60))
        agent = first_index_agent_factory(deck)
        with patch(
            "evaluation.match_runner.to_observation_class",
            return_value=SimpleNamespace(select=None),
        ):
            self.assertEqual(agent({}), deck)

    def test_is_deterministic(self) -> None:
        self.assertEqual(self.agent_for(2, 4, 6), self.agent_for(2, 4, 6))


if __name__ == "__main__":
    unittest.main()
