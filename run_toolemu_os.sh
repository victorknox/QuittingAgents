#!/bin/bash
#SBATCH --job-name=toolemu_os
#SBATCH --output=logs/exp_%x_%j.out
#SBATCH --error=logs/exp_%x_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --qos=default

# Exit on any error
set -e

# Print commands being executed
# set -x

# Usage: sbatch run_toolemu_os.sh <input_path> <agent_model> <simulator_model> <evaluator_model> <agent_type> <trunc_num> [additional_args]
INPUT_PATH=${1:-./assets/all_cases.json}
AGENT_MODEL=${2:-Qwen/Qwen2.5-1.5B-Instruct}
SIMULATOR_MODEL=${3:-Qwen/Qwen2.5-1.5B-Instruct}
EVALUATOR_MODEL=${4:-Qwen/Qwen2.5-1.5B-Instruct}
AGENT_TYPE=${5:-naive}
TRUNC_NUM=${6:-1000}
# Capture all additional arguments
ADDITIONAL_ARGS="${@:7}"

# Initialize conda properly for the batch job
eval "$(/nas/ucb/victorknox/software/miniconda3/bin/conda shell.bash hook)"
conda activate toolemu-latest || { echo "Failed to activate conda environment"; exit 1; }

# Change to the correct directory
cd /nas/ucb/victorknox/workspace/toolemu/ || { echo "Failed to change directory"; exit 1; }

# Create logs directory if it doesn't exist
mkdir -p logs

# Start GPU monitoring if GPUs are requested
if [ "${SLURM_GPUS_PER_NODE:-0}" -gt 0 ] || [ "${SLURM_JOB_GPUS:-0}" -gt 0 ] || grep -q -- '--gpus=[1-9]' <<< "$(head -n 10 $0)"; then
    (while true; do nvidia-smi >> logs/gpu_usage_${SLURM_JOB_ID}.log; sleep 30; done) &
    GPU_MON_PID=$!
fi

# 3. Run the evaluation
python scripts/run.py \
    --agent-model-name "$AGENT_MODEL" \
    --simulator-model-name "$SIMULATOR_MODEL" \
    --evaluator-model-name "$EVALUATOR_MODEL" \
    --input-path "$INPUT_PATH" \
    --agent-type "$AGENT_TYPE" \
    --auto \
    --track-costs \
    --trunc-num "$TRUNC_NUM" \
    -bs 1 \
    $ADDITIONAL_ARGS \
    || { echo "Evaluation failed"; exit 1; }

# Stop GPU monitoring if started
if [ ! -z "${GPU_MON_PID:-}" ]; then
    kill $GPU_MON_PID
fi

# 4. GPU usage monitoring (optional)
echo "Check GPU usage with: nvidia-smi > logs/gpu_usage_${SLURM_JOB_ID}.log"
nvidia-smi > logs/gpu_usage_${SLURM_JOB_ID}.log 