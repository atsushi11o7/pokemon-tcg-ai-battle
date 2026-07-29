"""Determinized MCTSの探索木本体(PUCTによる選択、ノード展開、バックプロパゲーション)。

隠れ情報は呼び出し側が`determinize.py`で1つの仮説に固定した上で`search_begin`を
呼んでおき、その結果得られる`SearchState`を根ノードとして木探索する。
公式サンプルコード(reinforcement-learning-and-mcts-sample-code.ipynb)のNode/Child/
create_nodeの構造を踏襲しつつ、局面の評価は外部から`eval_fn`として受け取れる形にしている
(公式サンプルの`eval_nn`に相当。方策の事前分布と価値を1回の呼び出しでまとめて返す)。
"""

import itertools
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SAMPLE_SUBMISSION_DIR = ROOT / "data" / "sample_submission" / "sample_submission"
sys.path.insert(0, str(SAMPLE_SUBMISSION_DIR))

from cg.api import search_step  # noqa: E402

MAX_ACTIONS_PER_NODE = 64  # 多重選択(maxCount>1)での組み合わせ爆発を抑えるための上限
PUCT_C = 0.4  # 公式サンプルコードと同じ探索係数


class Child:
    """MCTSノードの子(まだ展開されていない場合は`node`がNone)。"""

    def __init__(self, select: list[int], prob: float) -> None:
        """
        Args:
            select: この子が対応する選択(`search_step`にそのまま渡すindexのリスト)。
            prob: 方策の事前分布におけるこの選択の確率(PUCTのprior)。
        """
        self.node: Node | None = None
        self.select = select
        self.prob = prob


class Node:
    """MCTS探索木のノード。1つの`SearchState`(=1つの選択局面)に対応する。"""

    def __init__(self, parent: "Node | None", state) -> None:
        """
        Args:
            parent: 親ノード(根ノードの場合はNone)。
            state: このノードに対応する`SearchState`。
        """
        self.total = 0.0  # このノードを訪問した際のvalueの合計(バックプロパゲーションで累積)
        self.visit = 0
        self.parent = parent
        self.children: list[Child] = []
        self.state = state

    def backprop(self, value: float) -> None:
        """このノードから根に向かって、valueを合計・訪問回数を加算していく。

        Args:
            value: このノードで得られた評価値、または終局結果(+1/-1/0)。
        """
        self.total += value
        self.visit += 1
        if self.parent is not None:
            self.parent.backprop(value)


def _enumerate_actions(select) -> list[list[int]]:
    """選択肢の中から、`minCount`以上`maxCount`以下の個数を選ぶ組み合わせを列挙する。

    ほとんどの選択は`minCount == maxCount == 1`の単一選択で、この場合は
    「各選択肢を1つずつ選ぶ」という単純な列挙になり、選択肢数nに対して線形(高々n件)
    にしかならないため打ち切らない(MAIN選択はATTACHのエネルギー×対象ポケモンの
    組み合わせなどで数十件になることがあるが、組み合わせ爆発ではないので問題無い)。
    2個以上を選ぶ複数選択は組み合わせ数が膨大になりうるため、`MAX_ACTIONS_PER_NODE`件で
    打ち切る(公式サンプルコードも同様に列挙件数の上限を設けている)。

    Args:
        select: 列挙対象の`SelectData`。

    Returns:
        list[list[int]]: 各要素が選択するindexのリスト(`search_step`にそのまま渡せる形式)。
    """
    n = len(select.option)
    actions: list[list[int]] = []
    for count in range(select.minCount, select.maxCount + 1):
        if count == 0:
            actions.append([])
        elif count == 1:
            actions.extend([i] for i in range(n))
        else:
            for combo in itertools.combinations(range(n), count):
                actions.append(list(combo))
                if len(actions) >= MAX_ACTIONS_PER_NODE:
                    return actions
    return actions


def create_node(parent: "Node | None", search_state, your_index: int, eval_fn) -> "Node":
    """`SearchState`から新しいノードを作る。

    終局していれば勝敗(+1/-1/0)を、していなければ`eval_fn`による評価値を、
    作成直後に即座に根までバックプロパゲーションする(公式サンプルコードと同じ扱い)。

    Args:
        parent: 親ノード(根ノードを作る場合はNone)。
        search_state: このノードのもとになる`SearchState`(`search_begin`/`search_step`の戻り値)。
        your_index: 探索の根本になっているプレイヤーのインデックス(勝敗の基準)。
        eval_fn: `(obs) -> (list[float] | None, float)`。`obs.current.yourIndex`視点での
            (合法な選択肢群に対する事前確率, 局面の価値)を返す関数。事前確率は
            単一選択(`minCount==maxCount==1`)以外の場合、またはNone、または列挙した
            行動数と長さが合わない場合は一様分布にフォールバックする。
            (公式サンプルコードの`eval_nn`に相当)。

    Returns:
        Node: 作成したノード。
    """
    node = Node(parent, search_state)
    obs = search_state.observation
    state = obs.current

    if state.result >= 0:
        if state.result == 2:
            value = 0.0
        elif state.result == your_index:
            value = 1.0
        else:
            value = -1.0
        node.backprop(value)
        return node

    actions = _enumerate_actions(obs.select)
    probs, value = eval_fn(obs)
    if (
        obs.select.minCount != 1
        or obs.select.maxCount != 1
        or probs is None
        or len(probs) != len(actions)
    ):
        probs = [1.0 / len(actions)] * len(actions)

    for action, prob in zip(actions, probs, strict=True):
        node.children.append(Child(action, prob))

    if state.yourIndex != your_index:
        value = -value
    node.backprop(value)
    return node


def _select_child(node: "Node", your_index: int) -> "Child":
    """PUCTスコアが最大の子を選ぶ。

    スコアは「その子(まだ展開されていなければ親ノード自身)の平均value」に、
    「事前確率が高く、まだ訪問回数が少ない子」ほど大きくなる探索ボーナスを加えたもの。
    平均valueは常に`your_index`視点(手番がyour_indexでないノードでは符号を反転)に
    揃えてから比較する。

    Args:
        node: 子を選ぶ対象のノード。
        your_index: 探索の根本になっているプレイヤーのインデックス。

    Returns:
        Child: 選ばれた子。
    """
    c = PUCT_C * math.sqrt(node.visit)
    best_child = None
    best_score = -math.inf
    for child in node.children:
        if child.node is None:
            avg_value = node.total / node.visit
            visit = 0
        else:
            avg_value = child.node.total / child.node.visit
            visit = child.node.visit
        if node.state.observation.current.yourIndex != your_index:
            avg_value = -avg_value
        score = avg_value + c * child.prob / (1 + visit)
        if score > best_score:
            best_score = score
            best_child = child
    return best_child


def run_mcts(
    root_state, your_index: int, eval_fn, search_count: int
) -> tuple[list[int], list[float], float]:
    """根の`SearchState`からPUCTで`search_count`回のシミュレーションを行う。

    Args:
        root_state: 探索の起点になる`SearchState`(通常は`search_begin`の戻り値)。
        your_index: 探索の根本になっているプレイヤーのインデックス(勝敗の基準)。
        eval_fn: `create_node`に渡す局面評価関数。
        search_count: シミュレーション回数(MCTSのイテレーション数)。

    Returns:
        tuple[list[int], list[float], float]: (select, policy_target, root_value)。
            select: 根から見て最も訪問回数の多い子の選択(`search_step`にそのまま渡せる形式)。
            policy_target: `root.children`と同じ順序に並んだ、訪問回数を正規化した分布
                (自己対戦の学習で方策の教師信号として使う)。
            root_value: 探索全体で洗練された根ノードの平均value(`your_index`視点。
                自己対戦の学習で価値の教師信号のもとになる)。
    """
    root = create_node(None, root_state, your_index, eval_fn)

    for _ in range(search_count):
        current = root
        while True:
            child = _select_child(current, your_index)
            if child.node is None:
                next_state = search_step(current.state.searchId, child.select)
                child.node = create_node(current, next_state, your_index, eval_fn)
                break
            current = child.node
            if current.state.observation.current.result >= 0:
                # 既に終局しているノードを再訪問した場合、その結果を再度加算する
                current.backprop(current.total / current.visit)
                break

    root_value = root.total / root.visit
    visits = [child.node.visit if child.node is not None else 0 for child in root.children]
    total_visits = sum(visits)
    if total_visits == 0:
        # 1回もシミュレーションが展開されなかった場合(search_count=0等)のフォールバック
        policy_target = [1.0 / len(root.children)] * len(root.children)
        return root.children[0].select, policy_target, root_value

    policy_target = [v / total_visits for v in visits]
    best_index = max(range(len(root.children)), key=lambda i: visits[i])
    return root.children[best_index].select, policy_target, root_value
