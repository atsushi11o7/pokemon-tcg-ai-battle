"""`PolicyValueNet`のネットワーク構成。PPO/MCTSで共有し、チェックポイントを
相互に使い回せるようにする(定義箇所を1つに保つことで、サイズを変えるときの修正漏れを防ぐ)。
"""

# カードIDのone-hotをどう持つか。詳細は`sparse_features.py`のレイアウト定義を参照。
#   "per_role"    … 出現箇所ごとに独立したカード表(入力96,860次元)
#   "shared_card" … カード表を1つ共有し、出現箇所は役割埋め込みで区別(入力約3,000次元)
# 学習時と推論時で食い違うと`load_state_dict`が形状不一致で落ちるため取り違えは起きない。
FEATURE_LAYOUT = "shared_card"

# 容量診断では当てはめ残差がほぼ0で、表現力は律速ではなかった。一方MCTSラップ推論は
# 探索回数がそのまま強さに効くため、余った容量は探索深さへ回す方が期待値が高い。
# 層のパラメータはD_MODELの二乗で効くので、256→128で計算量はおよそ1/4になる。
D_MODEL = 128
NUM_HEADS = 4
D_FEEDFORWARD = 512
NUM_LAYERS_ENCODER = 3
NUM_LAYERS_DECODER = 3
