"""`PolicyValueNet`のネットワーク構成。PPO/MCTSで共有する。"""

# 手札を表すトークン数。カードIDの昇順に1枚ずつ別トークンへ置き、
# 溢れた分は最後のトークンへ合算する。1なら全カードを1トークンに合算する。
HAND_TOKENS = 12

# ベンチの枠数。
BENCH_SLOTS = 5

D_MODEL = 128
NUM_HEADS = 4
D_FEEDFORWARD = 1024
NUM_LAYERS_ENCODER = 4
NUM_LAYERS_DECODER = 4

DROPOUT = 0.1
