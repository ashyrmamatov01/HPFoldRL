#!/usr/bin/env bash
# 6-job matrix: {tabular-Q, DQN-MLP, DQN-CNN} × {20-mer, 12-mer}

set -e
BASE="runs"                 # common root for all artefacts

SEQ_LONG="HPHPPHHPHPPHPHHPPHPH"   # length-20 benchmark
SEQ_SHORT="HPHPPHHPHPPH"          # length-12 toy

# ------------------------------- #
# 1)     TABULAR-Q(λ) + UCB       #
# ------------------------------- #
# python -m hp_problem.scripts.train_tabular_q \
#   --sequence "$SEQ_LONG"          \
#   --episodes 120000               \
#   --exploration ucb --ucb-c 1.4   \
#   --alpha 0.6  --alpha-decay      \
#   --lam 0.9                       \
#   --log-interval 2000             \
#   --outdir "$BASE/tabular_long" &

# python -m hp_problem.scripts.train_tabular_q \
#   --sequence "$SEQ_SHORT"         \
#   --episodes 80000                \
#   --exploration ucb --ucb-c 1.4   \
#   --alpha 0.6  --alpha-decay      \
#   --lam 0.9                       \
#   --log-interval 2000             \
#   --outdir "$BASE/tabular_short" &

# ------------------------------- #
# 2)     DQN – MLP (Double⁺Duel)  #
# ------------------------------- #
# python -m hp_problem.scripts.train_dqn \
#   --sequence "$SEQ_LONG"          \
#   --episodes 250000               \
#   --network-type mlp              \
#   --hidden-dims 512 512           \
#   --batch-size 128 --buffer-size 80000 \
#   --gamma 0.995 --lr 3e-4         \
#   --eps-start 1.0 --eps-end 0.05  \
#   --target-update-interval 2000   \
#   --outdir "$BASE/dqn_mlp_long"   \
#   --device cuda &

# python -m hp_problem.scripts.train_dqn \
#   --sequence "$SEQ_SHORT"         \
#   --episodes 150000               \
#   --network-type mlp              \
#   --hidden-dims 512 512           \
#   --batch-size 128 --buffer-size 60000 \
#   --gamma 0.995 --lr 3e-4         \
#   --eps-start 1.0 --eps-end 0.05  \
#   --target-update-interval 1500   \
#   --outdir "$BASE/dqn_mlp_short"  \
#   --device cuda &

# ------------------------------- #
# 3)   DQN – CNN + NoisyNet        #
# ------------------------------- #
# python -m hp_problem.scripts.train_dqn \
#   --sequence "$SEQ_LONG"          \
#   --episodes 250000               \
#   --network-type cnn --board-size 21 \
#   --cnn-hidden 256 256            \
#   --batch-size 128 --buffer-size 80000 \
#   --gamma 0.995 --lr 2.5e-4       \
#   --eps-start 0.2 --eps-end 0.01  \
#   --target-update-interval 2000   \
#   --outdir "$BASE/dqn_cnn_long"   \
#   --device cuda &

# python -m hp_problem.scripts.train_dqn \
#   --sequence "$SEQ_SHORT"         \
#   --episodes 150000               \
#   --network-type cnn --board-size 15 \
#   --cnn-hidden 256 256            \
#   --batch-size 128 --buffer-size 60000 \
#   --gamma 0.995 --lr 2.5e-4       \
#   --eps-start 0.2 --eps-end 0.01  \
#   --target-update-interval 1500   \
#   --outdir "$BASE/dqn_cnn_short"  \
#   --device cuda &

# add another attn model
python -m hp_problem.scripts.train_dqn \
  --sequence "$SEQ_LONG"          \
  --episodes 250000               \
  --network-type attn --board-size 21 \
  --cnn-hidden 256 256            \
  --batch-size 1024 --buffer-size 80000 \
  --gamma 0.995 --lr 2.5e-4       \
  --eps-start 0.2 --eps-end 0.01  \
  --target-update-interval 2000   \
  --outdir "$BASE/dqn_attn_long"   \
  --device cuda &

python -m hp_problem.scripts.train_dqn \
  --sequence "$SEQ_SHORT"         \
  --episodes 150000               \
  --network-type attn --board-size 15 \
  --cnn-hidden 256 256            \
  --batch-size 2048 --buffer-size 60000 \
  --gamma 0.995 --lr 2.5e-4       \
  --eps-start 0.2 --eps-end 0.01  \
  --target-update-interval 1500   \
  --outdir "$BASE/dqn_attn_short"  \
  --device cuda &
wait
echo "All six jobs launched ✔"


