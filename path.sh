export PYTHONNOUSERSITE=1
# Avoid making training depend on the W&B background service. Set WANDB_MODE
# explicitly to "offline" or "online" before running to enable tracking.
export WANDB_MODE=${WANDB_MODE:-disabled}
#export PYTHONPATH="."

eval "$(conda shell.bash hook)"
# eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"
conda activate qwen3-asr
