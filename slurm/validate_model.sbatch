#!/usr/bin/env bash
#SBATCH --job-name=cal-agency-live
#SBATCH --partition=gpu
#SBATCH --account=idealab
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100.40gb:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=logs/validate-%j.out
#SBATCH --error=logs/validate-%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR}"
CONTAINER="${ROOT}/containers/vllm-openai-v0.26.0.sif"
MODEL_KEY="${MODEL_KEY:-qwen}"
PORT="${PORT:-18120}"
case "${MODEL_KEY}" in
  qwen) MODEL="Qwen/Qwen3.5-4B" ;;
  llama) MODEL="unsloth/Llama-3.1-8B-Instruct" ;;
  *) echo "unknown MODEL_KEY=${MODEL_KEY}" >&2; exit 2 ;;
esac
RESULTS="${ROOT}/results/${SLURM_JOB_ID}-${MODEL_KEY}"

export HF_HOME="${HOME}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export SINGULARITYENV_HF_HOME="${HF_HOME}"
export SINGULARITYENV_HF_HUB_CACHE="${HF_HUB_CACHE}"
export SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_TOKENIZERS_PARALLELISM=false
export SINGULARITYENV_PYTHONPATH="${ROOT}/src"
mkdir -p "${ROOT}/logs" "${RESULTS}"

if [[ ! -s "${CONTAINER}" ]]; then
  echo "missing ${CONTAINER}; run slurm/setup_runtime.sbatch" >&2
  exit 2
fi

cleanup() {
  status=$?
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  echo "[$(date -Is)] cleanup status=${status}"
  exit "${status}"
}
trap cleanup EXIT INT TERM

run_container() {
  singularity exec --nv --bind "${ROOT}:${ROOT}" --pwd "${ROOT}" "${CONTAINER}" "$@"
}

{
  echo "timestamp=$(date -Is)"
  echo "job=${SLURM_JOB_ID}"
  echo "host=$(hostname)"
  echo "model=${MODEL}"
  echo "model_key=${MODEL_KEY}"
  echo "git_commit=$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null || echo uncommitted)"
  echo "git_status=$(git -C "${ROOT}" status --porcelain 2>/dev/null | base64 -w0 || true)"
} >"${RESULTS}/run.env"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv \
  | tee "${RESULTS}/gpus.csv"

CUDA_VISIBLE_DEVICES=0 run_container \
  vllm serve "${MODEL}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --served-model-name "${MODEL}" \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.88 \
    --enforce-eager \
    --generation-config vllm \
    --logprobs-mode raw_logprobs \
    >"${RESULTS}/server.log" 2>&1 &
SERVER_PID=$!

SERVER_READY=0
for _ in $(seq 1 180); do
  if curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "[$(date -Is)] API server ready"
    SERVER_READY=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "API server exited during startup" >&2
    tail -200 "${RESULTS}/server.log" >&2
    exit 1
  fi
  sleep 5
done
if [[ "${SERVER_READY}" != 1 ]]; then
  echo "timed out waiting for API server" >&2
  tail -200 "${RESULTS}/server.log" >&2 || true
  exit 1
fi
curl -fsS "http://127.0.0.1:${PORT}/v1/models" >"${RESULTS}/models.json"

CUDA_VISIBLE_DEVICES=1 run_container python3 scripts/live_backend.py \
  --mode vllm-native \
  --model "${MODEL}" \
  --output "${RESULTS}/vllm-native.json" \
  >"${RESULTS}/vllm-native.log" 2>&1

CUDA_VISIBLE_DEVICES=1 run_container python3 scripts/live_backend.py \
  --mode transformers \
  --model "${MODEL}" \
  --output "${RESULTS}/transformers.json" \
  >"${RESULTS}/transformers.log" 2>&1

CUDA_VISIBLE_DEVICES=0 run_container python3 scripts/live_backend.py \
  --mode api \
  --model "${MODEL}" \
  --base-url "http://127.0.0.1:${PORT}" \
  --output "${RESULTS}/api.json" \
  >"${RESULTS}/api.log" 2>&1

run_container python3 scripts/compare_live.py \
  --api "${RESULTS}/api.json" \
  --vllm-native "${RESULTS}/vllm-native.json" \
  --transformers "${RESULTS}/transformers.json" \
  --output "${RESULTS}/summary.json" \
  --report "${RESULTS}/REPORT.md" \
  --strict

run_container python3 - <<'PY' >"${RESULTS}/versions.json"
import importlib.metadata as metadata
import json
import platform

names = ["calibrated-agency", "vllm", "transformers", "torch", "tokenizers"]
versions = {}
for name in names:
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps({
    "python": platform.python_version(),
    "packages": versions,
}, indent=2))
PY

sha256sum "${RESULTS}"/*.json "${RESULTS}/REPORT.md" >"${RESULTS}/SHA256SUMS"
cat "${RESULTS}/REPORT.md"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
