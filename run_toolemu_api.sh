#!/bin/bash
#SBATCH --job-name=toolemu_api
#SBATCH --output=logs/exp_%x_%j.out
#SBATCH --error=logs/exp_%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
#SBATCH --qos=default

# Exit on any error
set -e

# Print commands being executed
# set -x

# Disable torch.compile for vLLM to avoid compilation errors with Qwen models
export TORCH_COMPILE_DISABLE=1

# Usage: sbatch run_toolemu_api.sh <input_path> <agent_model> <simulator_model> <evaluator_model> <agent_type> <trunc_num>
INPUT_PATH=${1:-./assets/all_cases.json}
AGENT_MODEL=${2:-gpt-4o-mini}
SIMULATOR_MODEL=${3:-gpt-4o-mini}
EVALUATOR_MODEL=${4:-gpt-4o-mini}
AGENT_TYPE=${5:-naive}
TRUNC_NUM=${6:-1000}

# Validate required parameters
if [ -z "$TRUNC_NUM" ]; then
    echo "Error: trunc_num is required"
    echo "Usage: sbatch run_toolemu_api.sh <input_path> <agent_model> <simulator_model> <evaluator_model> <agent_type> <trunc_num>"
    exit 1
fi

# This script is for API-based models (OpenAI, Anthropic, etc.).
# Adjust --batch-size below for best throughput without hitting rate limits.

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

# Run the evaluation
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
    || { echo "Evaluation failed"; exit 1; } 

# Stop GPU monitoring if started
if [ ! -z "${GPU_MON_PID:-}" ]; then
    kill $GPU_MON_PID
fi 