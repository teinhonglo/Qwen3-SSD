export PYTHONNOUSERSITE=1
# Keep experiment tracking available while defaulting to a network-free run.
export WANDB_MODE=${WANDB_MODE:-offline}
#export PYTHONPATH="."

eval "$(conda shell.bash hook)"
# eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"
conda activate qwen3-asr
