"""Determinized MCTSの探索木本体(PUCTによる選択、ノード展開、バックプロパゲーション)。

呼び出し側が`determinize.py`で隠れ情報を1つの仮説に固定した上で`search_begin`を呼び、
その結果得られる`SearchState`を根ノードとして木探索する。局面の評価(事前確率・価値)は
外部から`eval_fn`として受け取る。
"""

import itertools
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SAMPLE_SUBMISSION_DIR = ROOT / "data" / "sample_submission" / "sample_submission"
sys.path.insert(0, str(SAMPLE_SUBMISSION_DIR))

from cg.api import search_release, search_step  # noqa: E402

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


def _unrank_combination(n: int, count: int, rank: int) -> list[int]:
    """辞書順の組み合わせを全列挙せず、rank番目だけを取り出す。"""
    result: list[int] = []
    start = 0
    for position in range(count):
        remaining = count - position - 1
        for candidate in range(start, n):
            suffixes = math.comb(n - candidate - 1, remaining)
            if rank < suffixes:
                result.append(candidate)
                start = candidate + 1
                break
            rank -= suffixes
    return result


def _enumerate_actions(select) -> list[list[int]]:
    """合法な複数選択を列挙し、多すぎる場合はrank空間から均等に採る。"""
    n = len(select.option)
    counts = list(range(select.minCount, select.maxCount + 1))
    sizes = [math.comb(n, count) for count in counts]
    total = sum(sizes)
    if total <= MAX_ACTIONS_PER_NODE:
        return [
            list(combo) for count in counts for combo in itertools.combinations(range(n), count)
        ]

    # 先頭だけを採ると後方indexを含む合法手が消えるため、全rankから等間隔に選ぶ。
    ranks = sorted(
        {round(i * (total - 1) / (MAX_ACTIONS_PER_NODE - 1)) for i in range(MAX_ACTIONS_PER_NODE)}
    )
    actions: list[list[int]] = []
    for rank in ranks:
        for count, size in zip(counts, sizes, strict=True):
            if rank < size:
                actions.append(_unrank_combination(n, count, rank))
                break
            rank -= size
    return actions


def create_node(parent: "Node | None", search_state, your_index: int, eval_fn) -> "Node":
    """`SearchState`から新しいノードを作る。

    終局していれば勝敗(+1/-1/0)を、していなければ`eval_fn`による評価値を、
    作成直後に即座に根までバックプロパゲーションする(公式サンプルコードと同じ扱い)。

    Args:
        parent: 親ノード(根ノードを作る場合はNone)。
        search_state: このノードのもとになる`SearchState`(`search_begin`/`search_step`の戻り値)。
        your_index: 探索の根本になっているプレイヤーのインデックス(勝敗の基準)。
        eval_fn: `(obs, actions) -> (list[float] | None, float)`。`obs.current.yourIndex`
            視点での(列挙済み行動`actions`に対する事前確率, 局面の価値)を返す関数。
            事前確率がNone、または列挙した行動数と長さが合わない場合は一様分布に
            フォールバックする。(公式サンプルコードの`eval_nn`に相当。行動群を渡すのは、
            疎な特徴量エンコーディング(`sparse_features.get_decoder_input`)が
            列挙済みの行動そのものを必要とするため)。

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
    probs, value = eval_fn(obs, actions)
    if probs is None or len(probs) != len(actions):
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
    if not node.children:
        # `_enumerate_actions`が空を返した(例: minCount/maxCountに対して選択肢数が
        # 足りない、といった不整合なSelectData)場合にここへ来る。Noneを黙って返すと
        # 呼び出し側が`child.node`で意味不明なAttributeErrorになるため、ここで
        # 原因が分かる形の例外にしておく(呼び出し側の1試合単位の例外キャッチで
        # 引き続き吸収される)。
        raise RuntimeError(
            "_select_child called on a node with no enumerated actions "
            f"(degenerate SelectData?); state={node.state.observation.current!r}"
        )

    c = PUCT_C * math.sqrt(node.visit)
    best_child = None
    best_score = -math.inf
    for child in node.children:
        if child.node is None:
            # 未展開の辺には価値観測がない。親のQをコピーすると、親が高評価なだけで
            # 未知の全行動まで高評価になり、priorによる探索が歪む。
            avg_value = 0.0
            visit = 0
        else:
            avg_value = child.node.total / child.node.visit
            visit = child.node.visit
        if node.state.observation.current.yourIndex != your_index:
            avg_value = -avg_value
        score = avg_value + c * child.prob / (1 + visit)
        # NaNは比較が常にFalseになるため、素通しすると`best_child`がNoneのまま返り、
        # 呼び出し側が原因の分からないAttributeErrorで落ちる。NaNの子は選ばないだけに留め、
        # 全部がNaN(=ネットワーク出力が壊れている)のときだけ下で明示的に落とす。
        if math.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_child = child
    if best_child is None:
        raise RuntimeError(
            "all PUCT scores are NaN; the policy network likely produced NaN priors "
            f"(children={len(node.children)})"
        )
    return best_child


def run_mcts(
    root_state,
    your_index: int,
    eval_fn,
    search_count: int,
    root_dirichlet_alpha: float | None = None,
    root_noise_fraction: float = 0.0,
) -> tuple[list[int], list[float], float, list[list[int]]]:
    """根の`SearchState`からPUCTで`search_count`回のシミュレーションを行う。

    Args:
        root_state: 探索の起点になる`SearchState`(通常は`search_begin`の戻り値)。
        your_index: 探索の根本になっているプレイヤーのインデックス(勝敗の基準)。
        eval_fn: `create_node`に渡す局面評価関数。
        search_count: シミュレーション回数(MCTSのイテレーション数)。

    Returns:
        tuple[list[int], list[float], float, list[list[int]]]:
            (select, policy_target, root_value, actions)。
            select: 根から見て最も訪問回数の多い子の選択(`search_step`にそのまま渡せる形式)。
            policy_target: `actions`と同じ順序に並んだ、訪問回数を正規化した分布
                (自己対戦の学習で方策の教師信号として使う)。
            root_value: 探索全体で洗練された根ノードの平均value(`your_index`視点。
                自己対戦の学習で価値の教師信号のもとになる)。
            actions: 根で列挙された行動一覧(`policy_target`と同じ順序。学習時に
                `sparse_features.get_decoder_input`へそのまま渡せる)。
    """
    root = create_node(None, root_state, your_index, eval_fn)
    actions = [child.select for child in root.children]
    if root_dirichlet_alpha is not None and root_noise_fraction > 0 and len(root.children) > 1:
        noise = [random.gammavariate(root_dirichlet_alpha, 1.0) for _ in root.children]
        noise_total = sum(noise)
        for child, sample in zip(root.children, noise, strict=True):
            child.prob = (
                1.0 - root_noise_fraction
            ) * child.prob + root_noise_fraction * sample / noise_total
    # search_stepは呼ぶたびに新しいSearchStateをエンジン側に確保するので、使い終わったら
    # search_releaseで明示的に解放する(放置するとネイティブメモリがリークする)。
    # 根ノード(root_state)はsearch_begin/search_endが管理するのでここでは対象外。
    created_search_ids: list[int] = []

    try:
        for _ in range(search_count):
            current = root
            while True:
                child = _select_child(current, your_index)
                if child.node is None:
                    next_state = search_step(current.state.searchId, child.select)
                    created_search_ids.append(next_state.searchId)
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
            return root.children[0].select, policy_target, root_value, actions

        policy_target = [v / total_visits for v in visits]
        best_index = max(range(len(root.children)), key=lambda i: visits[i])
        return root.children[best_index].select, policy_target, root_value, actions
    finally:
        for search_id in created_search_ids:
            search_release(search_id)
