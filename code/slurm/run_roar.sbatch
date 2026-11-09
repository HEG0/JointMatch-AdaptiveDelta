#!/bin/bash
# SLURM job array for Penn State Roar: 2 variants x 5 seeds = 10 independent single-GPU jobs.
#   task 0-4 -> adaptive_ema seeds 0-4
#   task 5-9 -> paper_fixed  seeds 0-4   (control; delete this variant and use --array=0-4 to skip it)
#
# Fill in the <PLACEHOLDERS>, then from the repo root:   mkdir -p slurm_logs && sbatch code/slurm/run_roar.sbatch
# Check the account / partition / GPU syntax against the current Roar documentation.
#
#SBATCH --job-name=jm-adaptive-delta
#SBATCH --account=<YOUR_ALLOCATION_OR_ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --array=0-9
#SBATCH --output=slurm_logs/%x_%A_%a.out

set -e

REPO_DIR=${REPO_DIR:-$SLURM_SUBMIT_DIR}          # path to this repo (default: where you ran sbatch)
OUT_DIR=${OUT_DIR:-$REPO_DIR/code/experiment/ag_news}   # put this on scratch/work storage if your home quota is small

# ---- environment: edit to match how you installed it ----
# module load anaconda3
source ~/.bashrc
conda activate jointmatch

# models / nltk data were cached beforehand by `python code/setup_offline.py` on the login node
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

VARIANTS=(adaptive_ema paper_fixed)
NSEEDS=5
VARIANT=${VARIANTS[$((SLURM_ARRAY_TASK_ID / NSEEDS))]}
SEED=$((SLURM_ARRAY_TASK_ID % NSEEDS))
echo "variant=$VARIANT seed=$SEED host=$(hostname)"

cd "$REPO_DIR/code"
python panel_main.py --variant "$VARIANT" --seeds "$SEED" --experiment_home "$OUT_DIR" --device_idx 0
