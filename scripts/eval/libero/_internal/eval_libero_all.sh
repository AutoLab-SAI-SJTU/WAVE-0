#!/usr/bin/env bash
# Multi-GPU original LIBERO launcher for WAVE-0.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

case ${1:-} in
  --help|-h)
    echo 'Usage: POLICY_PATH=/path/to/wave0 bash eval_libero_all.sh [--dry-run]'
    echo 'Defaults: 8 GPUs x 6 workers, 40 tasks x 50 episodes, seed 42.'
    exit 0
    ;;
  ''|--dry-run) ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac

POLICY_PATH=${POLICY_PATH:?Set POLICY_PATH.}
PYTHON=${PYTHON:-python}
TASKS_JSON=${TASKS_JSON:-${SCRIPT_DIR}/original_libero_tasks.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/outputs/eval/libero}
RUN_ID=${RUN_ID:-chunk32_warmup1000_ckpt10k_$(date +%Y%m%d_%H%M%S)}
NUM_GPUS=${NUM_GPUS:-8}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-6}
N_EPISODES=${N_EPISODES:-50}
SEED=${SEED:-42}
MAX_TASKS=${MAX_TASKS:-0}
TASK_TIMEOUT=${TASK_TIMEOUT:-43200}
STARTUP_STAGGER_SECONDS=${STARTUP_STAGGER_SECONDS:-1}
RESUME=${RESUME:-0}
WORLD_MODEL_STAGE=${WORLD_MODEL_STAGE:-inference}
COMPILE_MODEL=${COMPILE_MODEL:-false}
N_ACTION_STEPS=${N_ACTION_STEPS:-10}

for name in NUM_GPUS WORKERS_PER_GPU N_EPISODES; do
  [[ ${!name} =~ ^[1-9][0-9]*$ ]] || { echo "${name} must be positive" >&2; exit 2; }
done
[[ ${MAX_TASKS} =~ ^[0-9]+$ ]] || { echo "MAX_TASKS must be non-negative" >&2; exit 2; }
[[ ${TASK_TIMEOUT} =~ ^[1-9][0-9]*$ ]] || { echo "TASK_TIMEOUT must be positive" >&2; exit 2; }
[[ ${RESUME} == 0 || ${RESUME} == 1 ]] || { echo "RESUME must be 0 or 1" >&2; exit 2; }
[[ -f ${TASKS_JSON} ]] || { echo "Tasks JSON not found: ${TASKS_JSON}" >&2; exit 1; }
command -v flock >/dev/null || { echo "Missing flock" >&2; exit 1; }
command -v setsid >/dev/null || { echo "Missing setsid" >&2; exit 1; }
command -v timeout >/dev/null || { echo "Missing timeout" >&2; exit 1; }

IFS=',' read -r -a GPU_LIST <<< "${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS - 1)))}}"
(( ${#GPU_LIST[@]} >= NUM_GPUS )) || { echo "Not enough GPUs in GPU_LIST" >&2; exit 1; }
GPU_LIST=("${GPU_LIST[@]:0:NUM_GPUS}")
TOTAL_WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))
RUN_ROOT=${OUTPUT_ROOT}/${RUN_ID}
LOG_DIR=${RUN_ROOT}/logs
SHARD_DIR=${RUN_ROOT}/shards
STATUS_DIR=${RUN_ROOT}/status
WORKER_PY=${SCRIPT_DIR}/run_libero_worker.py
AGG_PY=${SCRIPT_DIR}/aggregate_libero_results.py

EXPECTED_TASKS=$(${PYTHON} - "${TASKS_JSON}" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
assert isinstance(rows, list) and rows
assert len({(r['suite'], int(r['task_id'])) for r in rows}) == len(rows)
print(len(rows))
PY
)
EXPECTED_RUN_TASKS=${EXPECTED_TASKS}
if (( MAX_TASKS > 0 )); then
  EXPECTED_RUN_TASKS=$((TOTAL_WORKERS * MAX_TASKS))
  (( EXPECTED_RUN_TASKS < EXPECTED_TASKS )) || EXPECTED_RUN_TASKS=${EXPECTED_TASKS}
fi

if [[ ${1:-} == --dry-run ]]; then
  echo "GPUs: ${GPU_LIST[*]} | workers/GPU: ${WORKERS_PER_GPU} | total workers: ${TOTAL_WORKERS}"
  echo "Tasks: ${EXPECTED_TASKS} (expected run tasks: ${EXPECTED_RUN_TASKS}) | episodes/task: ${N_EPISODES} | seed: ${SEED}"
  ${PYTHON} - "${TASKS_JSON}" "${MAX_TASKS}" "${TOTAL_WORKERS}" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
max_tasks, workers = int(sys.argv[2]), int(sys.argv[3])
for worker in range(workers):
    assigned = [r for i, r in enumerate(rows) if i % workers == worker]
    if max_tasks:
        assigned = assigned[:max_tasks]
    for row in assigned:
        print(f"worker={worker} {row['suite']}:{row['task_id']}")
PY
  exit 0
fi

mkdir -p "${LOG_DIR}" "${SHARD_DIR}" "${STATUS_DIR}"
if [[ -e ${RUN_ROOT}/run_config.json && ${RESUME} == 0 ]]; then
  echo "Run already exists: ${RUN_ROOT}; use a new RUN_ID or RESUME=1" >&2
  exit 1
fi
cat > "${RUN_ROOT}/run_config.json" <<EOF
{
  "benchmark": "libero_original",
  "policy_path": "${POLICY_PATH}",
  "tasks_json": "${TASKS_JSON}",
  "num_gpus": ${NUM_GPUS},
  "workers_per_gpu": ${WORKERS_PER_GPU},
  "n_episodes": ${N_EPISODES},
  "seed": ${SEED},
  "expected_tasks": ${EXPECTED_TASKS},
  "expected_run_tasks": ${EXPECTED_RUN_TASKS},
  "chunk_size": 32,
  "n_action_steps": ${N_ACTION_STEPS},
  "world_model_stage": "${WORLD_MODEL_STAGE}",
  "compile_model": "${COMPILE_MODEL}"
}
EOF

export PYTHONPATH="${REPO_ROOT}/src${LIBERO_SRC:+:${LIBERO_SRC}}${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}

printf '%s\n' "${GPU_LIST[@]}" > "${STATUS_DIR}/gpus.txt"
cp "${TASKS_JSON}" "${RUN_ROOT}/tasks.json"
: > "${STATUS_DIR}/workers.txt"
declare -a PIDS=()
cleanup() {
  trap - INT TERM
  for pid in "${PIDS[@]:-}"; do kill -TERM "${pid}" 2>/dev/null || true; done
  for pid in "${PIDS[@]:-}"; do wait "${pid}" 2>/dev/null || true; done
}
trap cleanup INT TERM

for ((worker=0; worker<TOTAL_WORKERS; worker++)); do
  gpu=${GPU_LIST[$((worker / WORKERS_PER_GPU))]}
  jsonl=${SHARD_DIR}/worker_$(printf '%02d' "${worker}").jsonl
  log=${LOG_DIR}/worker_$(printf '%02d' "${worker}").log
  (
    export CUDA_VISIBLE_DEVICES=${gpu}
    set +e
    setsid timeout --kill-after=10s "${TASK_TIMEOUT}" "${PYTHON}" "${WORKER_PY}" \
      --policy-path="${POLICY_PATH}" --tasks-json="${TASKS_JSON}" --output-jsonl="${jsonl}" \
      --worker-id="${worker}" --num-workers="${TOTAL_WORKERS}" --seed="${SEED}" \
      --n-episodes="${N_EPISODES}" --max-tasks="${MAX_TASKS}" \
      --world-model-stage="${WORLD_MODEL_STAGE}" --compile-model="${COMPILE_MODEL}" \
      --n-action-steps="${N_ACTION_STEPS}" \
      $([[ ${RESUME} == 1 ]] && echo --retry-failed) >"${log}" 2>&1
    rc=$?
    printf 'worker=%s gpu=%s rc=%s\n' "${worker}" "${gpu}" "${rc}" >> "${STATUS_DIR}/workers.txt"
    exit ${rc}
  ) &
  PIDS+=("$!")
  sleep "${STARTUP_STAGGER_SECONDS}"
done

fail=0
for pid in "${PIDS[@]}"; do wait "${pid}" || fail=1; done
trap - INT TERM
"${PYTHON}" "${AGG_PY}" --run-root="${RUN_ROOT}" --expected-tasks="${EXPECTED_RUN_TASKS}" | tee "${RUN_ROOT}/aggregate.log"
if (( fail )); then
  echo "One or more workers failed; see ${LOG_DIR}" >&2
  exit 1
fi
echo "DONE ${RUN_ROOT}"
