#!/usr/bin/env bash
# Evaluate one RoboTwin task with the reference WAVE-0 protocol.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
case ${1:-} in
    --help|-h)
        echo 'Usage: ROBOTWIN_ROOT=/path/to/RoboTwin POLICY_PATH=/path/to/checkpoint bash eval_robotwin.sh [--dry-run]'
        echo 'Options: TASK_NAME, TASK_CONFIG, N_EPISODES, SEED, POLICY_N_ACTION_STEPS, CUDA_VISIBLE_DEVICES, OUTPUT_DIR, PYTHON'
        exit 0 ;;
    ''|--dry-run) ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac

ROBOTWIN_ROOT=$(realpath -e -- "${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT to your RoboTwin checkout.}")
POLICY_PATH=$(realpath -e -- "${POLICY_PATH:?Set POLICY_PATH to a local WAVE-0 Stage-2 checkpoint.}")
PYTHON=$(command -v "${PYTHON:-python}")
[[ ${PYTHON} == /* ]] || PYTHON=${PWD}/${PYTHON}
TASK_NAME=${TASK_NAME:-place_empty_cup}
TASK_CONFIG=${TASK_CONFIG:-demo_clean}
N_EPISODES=${N_EPISODES:-100}
SEED=${SEED:-1000}
POLICY_N_ACTION_STEPS=${POLICY_N_ACTION_STEPS:-32}
MAX_EPISODES_RENDERED=${MAX_EPISODES_RENDERED:-0}
OUTPUT_DIR=$(realpath -m -- "${OUTPUT_DIR:-${REPO_ROOT}/outputs/eval/robotwin/${TASK_CONFIG}_${TASK_NAME}_$(date +%Y%m%d_%H%M%S)_$$}")
LOG_FILE=$(realpath -m -- "${LOG_FILE:-${OUTPUT_DIR}.log}")

[[ ${TASK_CONFIG} == demo_clean || ${TASK_CONFIG} == demo_randomized ]]
[[ ${N_EPISODES} =~ ^[1-9][0-9]*$ && ${POLICY_N_ACTION_STEPS} =~ ^[1-9][0-9]*$ ]]
[[ ${SEED} =~ ^(0|[1-9][0-9]*)$ && ${MAX_EPISODES_RENDERED} =~ ^(0|[1-9][0-9]*)$ ]]
for file in config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
    [[ -f ${POLICY_PATH}/${file} ]] || { echo "Missing checkpoint file: ${POLICY_PATH}/${file}" >&2; exit 1; }
done

# Read the official per-task limit; do not silently truncate long tasks at 400 steps.
EPISODE_LENGTH=$("${PYTHON}" - "${ROBOTWIN_ROOT}" "${POLICY_PATH}" "${TASK_NAME}" "${POLICY_N_ACTION_STEPS}" <<'PY'
import json
import re
import sys
from pathlib import Path
import yaml

root, checkpoint, task, steps = sys.argv[1:]
config = json.loads((Path(checkpoint) / "config.json").read_text())
if config["type"] != "wave0" or config.get("world_model_stage") not in {"stage2", "inference"}:
    raise SystemExit("Use a full WAVE-0 Stage-2 checkpoint; convert legacy weights with scripts/export_checkpoint.py.")
if int(steps) > config["chunk_size"]:
    raise SystemExit("POLICY_N_ACTION_STEPS exceeds the model chunk size.")
limits = yaml.safe_load((Path(root) / "task_config/_eval_step_limit.yml").read_text())
limit = limits.get(task)
if not re.fullmatch(r"[a-z][a-z0-9_]*", task) or not (Path(root) / "envs" / f"{task}.py").is_file() or type(limit) is not int or limit <= 0:
    raise SystemExit(f"Missing task or official positive step limit: {task}")
print(limit)
PY
)

export PYTHONPATH="${REPO_ROOT}/src:${ROBOTWIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export LEROBOT_ROBOTWIN_TASK_CONFIG=${TASK_CONFIG}
export LEROBOT_ROBOTWIN_OFFICIAL_INSTRUCTION=1 LEROBOT_ROBOTWIN_INSTRUCTION_TYPE=seen
export LEROBOT_ROBOTWIN_INSTRUCTION_MAX=100 LEROBOT_ROBOTWIN_CLEAR_CACHE_FREQ=1
export LEROBOT_ROBOTWIN_RENDER_ALL_CAMERAS=1
export LEROBOT_EVAL_MAX_EPISODES_RENDERED=${MAX_EPISODES_RENDERED}
export LEROBOT_ROBOTWIN_SKIP_OBS_WITHIN_CHUNK=$((MAX_EPISODES_RENDERED == 0))
unset LEROBOT_EVAL_EPISODE_SEEDS ROBOTWIN_FAIL_FAST RANK LOCAL_RANK WORLD_SIZE

RENAME_MAP='{"observation.images.head_camera":"observation.images.cam_high","observation.images.left_camera":"observation.images.cam_left_wrist","observation.images.right_camera":"observation.images.cam_right_wrist"}'
COMMAND=("${PYTHON}" -m lerobot.scripts.lerobot_eval
    --policy.path="${POLICY_PATH}"
    --policy.device=cuda --policy.dtype=bfloat16
    --policy.world_model_stage=inference --policy.compile_model=false
    --policy.n_action_steps="${POLICY_N_ACTION_STEPS}"
    --env.type=robotwin --env.task="${TASK_NAME}" --env.episode_length="${EPISODE_LENGTH}"
    --env.max_parallel_tasks=1 --eval.batch_size=1 --eval.use_async_envs=false
    --eval.n_episodes="${N_EPISODES}" --seed="${SEED}"
    --rename_map="${RENAME_MAP}" --output_dir="${OUTPUT_DIR}"
    --job_name="${TASK_CONFIG}_${TASK_NAME}")

if [[ ${1:-} == --dry-run ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%s TASK_CONFIG=%s max_steps=%s\n' "${CUDA_VISIBLE_DEVICES}" "${TASK_CONFIG}" "${EPISODE_LENGTH}"
    printf '%q ' "${COMMAND[@]}"; printf '\n'
    exit 0
fi
[[ ! -e ${OUTPUT_DIR}/eval_info.json ]] || { echo "Result already exists: ${OUTPUT_DIR}" >&2; exit 1; }
mkdir -p -- "${OUTPUT_DIR}" "$(dirname -- "${LOG_FILE}")"
cd -- "${ROBOTWIN_ROOT}"
{ printf '%q ' "${COMMAND[@]}"; printf '\n'; "${COMMAND[@]}"; } 2>&1 | tee -a "${LOG_FILE}"
