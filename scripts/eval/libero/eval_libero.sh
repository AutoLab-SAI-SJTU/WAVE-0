#!/usr/bin/env bash
# Original LIBERO evaluation for WAVE-0 chunk32 / h9 / WAM-warmup1000 checkpoint 10k.
# The checkpoint must already be exported to the public WAVE-0 `type=wave0` format.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

case ${1:-} in
  --help|-h)
    cat <<'EOF'
Usage:
  POLICY_PATH=/path/to/wave0/checkpoint \
  bash scripts/eval/libero/eval_libero.sh [--dry-run]

Defaults: NUM_GPUS=8 WORKERS_PER_GPU=6 N_EPISODES=50 SEED=42
Optional: GPU_LIST/CUDA_VISIBLE_DEVICES, LIBERO_SRC, LIBERO_CONFIG_PATH,
          OUTPUT_ROOT, RUN_ID, RESUME, TASKS_JSON, MAX_TASKS, TASK_TIMEOUT.
EOF
    exit 0
    ;;
  ''|--dry-run) ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac

POLICY_PATH=$(realpath -e -- "${POLICY_PATH:?Set POLICY_PATH to an exported WAVE-0 checkpoint.}")
PYTHON=${PYTHON:-python}
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 1
fi

N_EPISODES=${N_EPISODES:-50}
SEED=${SEED:-42}
NUM_GPUS=${NUM_GPUS:-8}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-6}
N_ACTION_STEPS=${N_ACTION_STEPS:-10}
WORLD_MODEL_STAGE=${WORLD_MODEL_STAGE:-inference}
COMPILE_MODEL=${COMPILE_MODEL:-false}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/outputs/eval/libero}
RUN_ID=${RUN_ID:-chunk32_warmup1000_ckpt10k_$(date +%Y%m%d_%H%M%S)}
TASKS_JSON=${TASKS_JSON:-${SCRIPT_DIR}/_internal/original_libero_tasks.json}

for name in N_EPISODES NUM_GPUS WORKERS_PER_GPU N_ACTION_STEPS; do
  [[ ${!name} =~ ^[1-9][0-9]*$ ]] || { echo "${name} must be positive" >&2; exit 2; }
done
[[ ${SEED} =~ ^[0-9]+$ ]] || { echo "SEED must be non-negative" >&2; exit 2; }
[[ ${WORLD_MODEL_STAGE} == inference ]] || { echo "WORLD_MODEL_STAGE must be inference" >&2; exit 2; }
[[ ${COMPILE_MODEL} == false ]] || { echo "COMPILE_MODEL must be false for this protocol" >&2; exit 2; }
[[ -f ${TASKS_JSON} ]] || { echo "Task catalog not found: ${TASKS_JSON}" >&2; exit 1; }

for file in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
  [[ -f ${POLICY_PATH}/${file} ]] || { echo "Missing checkpoint file: ${POLICY_PATH}/${file}" >&2; exit 1; }
done

"${PYTHON}" - "${POLICY_PATH}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config = json.loads((root / "config.json").read_text())
errors = []
if config.get("type") != "wave0":
    errors.append(f"config.type must be wave0, got {config.get('type')!r}")
if int(config.get("chunk_size", -1)) != 32:
    errors.append(f"chunk_size must be 32, got {config.get('chunk_size')!r}")
if int(config.get("n_action_steps", -1)) != 10:
    errors.append(f"n_action_steps must be 10, got {config.get('n_action_steps')!r}")
if config.get("world_model_stage") not in {"stage2", "inference"}:
    errors.append(f"checkpoint must be Stage-2/inference, got {config.get('world_model_stage')!r}")
if config.get("wam_loss_warmup_steps") not in {None, 1000}:
    errors.append(f"wam_loss_warmup_steps must be 1000, got {config.get('wam_loss_warmup_steps')!r}")
horizons = config.get("world_model_future_horizons")
if horizons is not None and list(horizons) != [9]:
    errors.append(f"world_model_future_horizons must be [9], got {horizons!r}")
queries = config.get("wam_latent_action_num_queries")
if queries is not None and int(queries) != 10:
    errors.append(f"wam_latent_action_num_queries must be 10, got {queries!r}")
for pipeline_name in ("policy_preprocessor.json", "policy_postprocessor.json"):
    data = json.loads((root / pipeline_name).read_text())
    for step in data.get("steps", []):
        state_file = step.get("state_file") or step.get("config", {}).get("state_file")
        if state_file and not (root / state_file).is_file():
            errors.append(f"{pipeline_name} references missing state file {state_file}")
if errors:
    raise SystemExit("Invalid WAVE-0 LIBERO checkpoint:\n- " + "\n- ".join(errors))
print("Validated WAVE-0 chunk32/h9 warmup1000 checkpoint:", root)
PY

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}
if [[ -n ${EXTRA_LIB_DIR:-} ]]; then
  [[ -d ${EXTRA_LIB_DIR} ]] || { echo "EXTRA_LIB_DIR is not a directory: ${EXTRA_LIB_DIR}" >&2; exit 1; }
  export LD_LIBRARY_PATH=${EXTRA_LIB_DIR}:${LD_LIBRARY_PATH:-}
fi
if [[ -n ${MAGICK_PRELOAD:-} ]]; then
  [[ -f ${MAGICK_PRELOAD} ]] || { echo "MAGICK_PRELOAD is not a file: ${MAGICK_PRELOAD}" >&2; exit 1; }
  export LD_PRELOAD=${MAGICK_PRELOAD}${LD_PRELOAD:+:${LD_PRELOAD}}
fi
if [[ -n ${LIBERO_CONFIG_PATH:-} ]]; then
  if [[ -f ${LIBERO_CONFIG_PATH} ]]; then
    LIBERO_CONFIG_PATH=$(dirname -- "${LIBERO_CONFIG_PATH}")
    export LIBERO_CONFIG_PATH
  fi
  [[ -f ${LIBERO_CONFIG_PATH}/config.yaml ]] || {
    echo "LIBERO_CONFIG_PATH does not contain config.yaml: ${LIBERO_CONFIG_PATH}" >&2; exit 1;
  }
fi

export POLICY_PATH OUTPUT_ROOT RUN_ID TASKS_JSON NUM_GPUS WORKERS_PER_GPU N_EPISODES SEED
export N_ACTION_STEPS WORLD_MODEL_STAGE COMPILE_MODEL PYTHON LIBERO_SRC=${LIBERO_SRC:-}
export RESUME=${RESUME:-0} MAX_TASKS=${MAX_TASKS:-0} TASK_TIMEOUT=${TASK_TIMEOUT:-43200}
export STARTUP_STAGGER_SECONDS=${STARTUP_STAGGER_SECONDS:-1}

if [[ ${1:-} == --dry-run ]]; then
  echo "policy=${POLICY_PATH}"
  echo "protocol=original-libero-40tasks episodes=${N_EPISODES} seed=${SEED} chunk=32 action_steps=${N_ACTION_STEPS} stage=${WORLD_MODEL_STAGE} compile=${COMPILE_MODEL}"
  bash "${SCRIPT_DIR}/_internal/eval_libero_all.sh" --dry-run
  exit 0
fi

exec bash "${SCRIPT_DIR}/_internal/eval_libero_all.sh"
