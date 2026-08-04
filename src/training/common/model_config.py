"""`PolicyValueNet`のネットワーク構成。BC/PPO/MCTSすべてで共有し、チェックポイントを
相互に使い回せるようにする(定義箇所を1つに保つことで、サイズを変えるときの修正漏れを防ぐ)。
"""

# generalist向けにさらに表現力を上げた構成(forward 1回1.58ms実測、無視できるコスト。
# state_dictは約107MBで提出サイズ上限197.7MiBに収まる)。
D_MODEL = 256
NUM_HEADS = 4
D_FEEDFORWARD = 512
NUM_LAYERS_ENCODER = 3
NUM_LAYERS_DECODER = 3
