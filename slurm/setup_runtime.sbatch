#!/usr/bin/env bash
#SBATCH --job-name=cal-agency-setup
#SBATCH --partition=hpc
#SBATCH --account=idealab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/setup-%j.out
#SBATCH --error=logs/setup-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR}"
CONTAINER_DIR="${ROOT}/containers"
CONTAINER="${CONTAINER_DIR}/vllm-openai-v0.26.0.sif"
export SINGULARITY_CACHEDIR="${HOME}/.cache/singularity"
export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${ROOT}/logs" "${CONTAINER_DIR}" "${SINGULARITY_CACHEDIR}" "${HF_HUB_CACHE}"

if [[ ! -s "${CONTAINER}" ]]; then
  echo "[$(date -Is)] pulling vLLM 0.26.0"
  singularity pull "${CONTAINER}" docker://vllm/vllm-openai:v0.26.0
fi

export SINGULARITYENV_HF_HOME="${HF_HOME}"
export SINGULARITYENV_HF_HUB_CACHE="${HF_HUB_CACHE}"
singularity exec "${CONTAINER}" python3 - <<'PY'
from huggingface_hub import snapshot_download

for model in ("Qwen/Qwen3.5-4B", "unsloth/Llama-3.1-8B-Instruct"):
    print(f"downloading {model}", flush=True)
    print(snapshot_download(model), flush=True)
PY

echo "[$(date -Is)] setup complete"
