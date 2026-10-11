#!/usr/bin/env bash
# Shared task queue: workers stay on one GPU and claim the next available task.
# Keep task logs and START/DONE/FAILED/LIVE output from the reference evaluations.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
case ${1:-} in
    --help|-h)
        echo 'Usage: ROBOTWIN_ROOT=/path/to/RoboTwin POLICY_PATH=/path/to/checkpoint bash eval_robotwin_all.sh [--dry-run]'
        echo 'Defaults: NUM_GPUS=8 WORKERS_PER_GPU=4 N_EPISODES=100 POLICY_N_ACTION_STEPS=32 SEED=1000'
        echo 'Options: GPU_LIST (or CUDA_VISIBLE_DEVICES), TASK_LIST, TASK_CONFIG_LIST, OUTPUT_ROOT, RUN_ID (or OUTPUT_DIR)'
        echo '         RESUME=0, MAX_EPISODES_RENDERED=0, STARTUP_STAGGER_SECONDS=1, TASK_TIMEOUT=43200, PYTHON'
        exit 0 ;;
    ''|--dry-run) ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac

ROBOTWIN_ROOT=$(realpath -e -- "${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT to your RoboTwin checkout.}")
POLICY_PATH=$(realpath -e -- "${POLICY_PATH:?Set POLICY_PATH to a local WAVE-0 Stage-2 checkpoint.}")
PYTHON=$(command -v "${PYTHON:-python}")
[[ ${PYTHON} == /* ]] || PYTHON=${PWD}/${PYTHON}
export ROBOTWIN_ROOT POLICY_PATH PYTHON
export N_EPISODES=${N_EPISODES:-100} SEED=${SEED:-1000}
export POLICY_N_ACTION_STEPS=${POLICY_N_ACTION_STEPS:-32} MAX_EPISODES_RENDERED=${MAX_EPISODES_RENDERED:-0}
NUM_GPUS=${NUM_GPUS:-8}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-4}
GPU_LIST=${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-}}
TASK_LIST=${TASK_LIST:-all}
TASK_CONFIG_LIST=${TASK_CONFIG_LIST:-demo_clean,demo_randomized}
STARTUP_STAGGER_SECONDS=${STARTUP_STAGGER_SECONDS:-1}
TASK_TIMEOUT=${TASK_TIMEOUT:-43200}
RESUME=${RESUME:-0}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/outputs/eval/robotwin}
RUN_ID=${RUN_ID:-wave0_$(date +%Y%m%d_%H%M%S)_$$}
RUN_ROOT=$(realpath -m -- "${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_ID}}")
LOG_DIR=${RUN_ROOT}/logs
STATUS_DIR=${RUN_ROOT}/status
ATTEMPT=$(date +%Y%m%d_%H%M%S)_$$

fail() { echo "Error: $*" >&2; exit 1; }
for name in NUM_GPUS WORKERS_PER_GPU N_EPISODES POLICY_N_ACTION_STEPS TASK_TIMEOUT; do
    [[ ${!name} =~ ^[1-9][0-9]*$ ]] || fail "${name} must be a positive integer."
done
[[ ${STARTUP_STAGGER_SECONDS} =~ ^[0-9]+([.][0-9]+)?$ ]] || fail 'Invalid STARTUP_STAGGER_SECONDS.'
case ${RESUME} in 0|false) RESUME=0 ;; 1|true) RESUME=1 ;; *) fail 'RESUME must be 0 or 1.' ;; esac
[[ ${RUN_ID} =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || fail 'RUN_ID must be a single directory name.'
for utility in flock setsid timeout; do command -v "${utility}" >/dev/null || fail "Missing ${utility}."; done
if [[ -n ${GPU_LIST} ]]; then
    [[ ${GPU_LIST} != ,* && ${GPU_LIST} != *, && ${GPU_LIST} != *,,* ]] || fail 'Empty GPU_LIST entry.'
    IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
else
    GPU_IDS=()
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do GPU_IDS+=("${gpu}"); done
fi
[[ ${#GPU_IDS[@]} == "${NUM_GPUS}" ]] || fail 'GPU_LIST/CUDA_VISIBLE_DEVICES must contain NUM_GPUS entries.'
declare -A SEEN_GPUS=()
for gpu in "${GPU_IDS[@]}"; do
    [[ ${gpu} =~ ^[a-zA-Z0-9-]+$ && -z ${SEEN_GPUS[${gpu}]:-} ]] || fail "Invalid/duplicate GPU: ${gpu}"
    SEEN_GPUS[${gpu}]=1
done

# Read the official 50-task list without importing torch or the simulator.
JOB_ROWS=$("${PYTHON}" - "${REPO_ROOT}" "${ROBOTWIN_ROOT}" "${TASK_LIST}" "${TASK_CONFIG_LIST}" <<'PY'
import ast
import sys
from pathlib import Path
import yaml

repo, root, tasks, configs = sys.argv[1:]
tree = ast.parse((Path(repo) / "src/lerobot/envs/robotwin.py").read_text())
official = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "ROBOTWIN_TASKS" for t in n.targets))
tasks = list(official) if tasks == "all" else tasks.split(",")
configs = configs.split(",")
if len(set(tasks)) != len(tasks) or not set(tasks) <= set(official):
    raise SystemExit("Unknown/duplicate task in TASK_LIST.")
if len(set(configs)) != len(configs) or not set(configs) <= {"demo_clean", "demo_randomized"}:
    raise SystemExit("Unknown/duplicate config in TASK_CONFIG_LIST.")
limits = yaml.safe_load((Path(root) / "task_config/_eval_step_limit.yml").read_text())
for config in configs:
    for task in tasks:
        if type(limits.get(task)) is not int or limits[task] <= 0 or not (Path(root) / "envs" / f"{task}.py").is_file():
            raise SystemExit(f"Missing task or official positive step limit: {task}")
        print(f"{config}:{task}")
PY
)
mapfile -t JOBS <<< "${JOB_ROWS}"
# Validate the checkpoint and print the exact command without loading a model.
TASK_CONFIG=${JOBS[0]%%:*} TASK_NAME=${JOBS[0]#*:} CUDA_VISIBLE_DEVICES=${GPU_IDS[0]} \
    OUTPUT_DIR=${RUN_ROOT}/command_preview bash "${SCRIPT_DIR}/eval_robotwin.sh" --dry-run
if [[ ${1:-} == --dry-run ]]; then
    printf 'GPUs: %s | workers/GPU: %s | output: %s\n' "${GPU_IDS[*]}" "${WORKERS_PER_GPU}" "${RUN_ROOT}"
    printf '%s\n' "${JOBS[@]}"
    exit 0
fi

mkdir -p -- "${RUN_ROOT}"
exec 8>"${RUN_ROOT}/.run.lock"
flock -n 8 || fail "Another evaluation is using ${RUN_ROOT}."
if [[ -f ${RUN_ROOT}/run_config.txt && ${RESUME} == 0 ]]; then
    fail 'Run already exists. Choose a new RUN_ID/OUTPUT_DIR, or set RESUME=1.'
fi
# Basic resume guard, not a code/model-content fingerprint. Keep weights and code unchanged.
RUN_CONFIG=$(printf '%s\n' "policy=${POLICY_PATH}" "robotwin=${ROBOTWIN_ROOT}" "episodes=${N_EPISODES}" \
    "seed=${SEED}" "action_steps=${POLICY_N_ACTION_STEPS}" "videos=${MAX_EPISODES_RENDERED}" "${JOB_ROWS}")
if [[ -f ${RUN_ROOT}/run_config.txt ]]; then
    cmp -s "${RUN_ROOT}/run_config.txt" <(printf '%s\n' "${RUN_CONFIG}") || fail 'Resume configuration differs from the saved run.'
fi
printf '%s\n' "${RUN_CONFIG}" >"${RUN_ROOT}/run_config.txt"
mkdir -p -- "${LOG_DIR}" "${STATUS_DIR}"
exec > >(tee -a "${RUN_ROOT}/launcher.log") 2>&1
printf '%s\n' "${JOB_ROWS}" >"${STATUS_DIR}/jobs.txt"

result_rate() {
    "${PYTHON}" - "$1/eval_info.json" "$2" "${N_EPISODES}" <<'PY'
import json
import math
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
n = int(sys.argv[3])
records = data["per_task"]
if len(records) != 1 or records[0]["task_group"] != sys.argv[2]:
    raise SystemExit("Wrong task result.")
metrics = records[0]["metrics"]
successes, seeds = metrics["successes"], metrics["seeds"]
if len(successes) != n or any(type(x) is not bool for x in successes):
    raise SystemExit("Incomplete episode results.")
if len(seeds) != n or any(type(x) is not int for x in seeds) or len(set(seeds)) != n:
    raise SystemExit("Missing/duplicate actual episode seeds.")
rate = 100 * sum(successes) / n
if data["overall"]["n_episodes"] != n or not math.isclose(data["overall"]["pc_success"], rate, abs_tol=1e-6):
    raise SystemExit("Aggregate does not match episode results.")
print(f"{rate:.6f}")
PY
}

summarize() {
    (
        flock -x 9
        "${PYTHON}" - "${RUN_ROOT}" "${N_EPISODES}" "${1:-live}" <<'PY'
import csv
import io
import json
import sys
from datetime import datetime
from pathlib import Path

root = Path(sys.argv[1])
fields = "config task gpu worker status exit_code success_rate output_dir log".split()
rows = [dict(zip(fields, (root / "status" / f"{job.replace(':', '_')}.tsv").read_text().strip().split("\t")))
        for job in (root / "status/jobs.txt").read_text().splitlines()]
for row in rows:
    row["success_rate"] = float(row["success_rate"]) if row["status"] == "OK" else ""
done = [r for r in rows if r["status"] == "OK"]
complete = len(done) == len(rows)
def mean(items):
    return sum(r["success_rate"] for r in items) / len(items) if items else None
def fmt(value):
    return "--" if value is None else f"{value:.2f}%"
summary = {"completed": len(done), "total": len(rows), "complete": complete,
           "failed": sum(r["status"] == "FAILED" for r in rows),
           "formal_benchmark": complete and len(rows) == 100 and int(sys.argv[2]) == 100,
           "completed_tasks_mean": mean(done), "overall_macro_success": mean(done) if complete else None,
           "per_config": {}}
for config in sorted({r["config"] for r in rows}):
    expected = [r for r in rows if r["config"] == config]
    finished = [r for r in done if r["config"] == config]
    summary["per_config"][config] = {"completed": len(finished), "total": len(expected),
        "completed_tasks_mean": mean(finished), "macro_success": mean(finished) if len(finished) == len(expected) else None}
message = (f"[LIVE {datetime.now():%Y-%m-%d %H:%M:%S}] "
           f"finished={len(done) + summary['failed']}/{len(rows)} ok={len(done)} failed={summary['failed']} | "
           f"overall={fmt(mean(done))} " + " ".join(
               f"{config.removeprefix('demo_')}={fmt(info['completed_tasks_mean'])}"
               for config, info in summary["per_config"].items()))
buffer = io.StringIO()
writer = csv.DictWriter(buffer, fieldnames=fields)
writer.writeheader()
writer.writerows(rows)
for name, content in (("summary.csv", buffer.getvalue()), ("summary.json", json.dumps(summary, indent=2) + "\n"),
                      ("live_summary.txt", message + "\n")):
    temporary = root / (name + ".tmp")
    temporary.write_text(content)
    temporary.replace(root / name)
print(message, flush=True)
if sys.argv[3] == "final":
    print(f"Completed: {len(done)}/{len(rows)} | Summary: {root / 'summary.csv'}", flush=True)
    raise SystemExit(0 if complete else 1)
PY
    ) 9>"${STATUS_DIR}/summary.lock"
}

PENDING=()
for job in "${JOBS[@]}"; do
    status_file=${STATUS_DIR}/${job/:/_}.tsv
    if [[ ${RESUME} == 1 && -f ${status_file} ]]; then
        IFS=$'\t' read -r config task gpu worker status code rate output log <"${status_file}"
        if [[ ${status} == OK && ${code} == 0 ]] && rate=$(result_rate "${output}" "${job#*:}" 2>/dev/null); then
            printf '%s\t%s\t%s\t%s\tOK\t0\t%s\t%s\t%s\n' "${config}" "${task}" "${gpu}" "${worker}" "${rate}" "${output}" "${log}" >"${status_file}"
            continue
        fi
    fi
    PENDING+=("${job}")
    printf '%s\t%s\t-\t-\tPENDING\t-\t-\t-\t-\n' "${job%%:*}" "${job#*:}" >"${status_file}"
done
QUEUE_FILE=${STATUS_DIR}/pending_jobs.txt
: >"${QUEUE_FILE}"
for job in "${PENDING[@]}"; do printf '%s\n' "${job}" >>"${QUEUE_FILE}"; done

claim_next_job() {
    local job=''
    (
        flock -x 9
        if IFS= read -r job <"${QUEUE_FILE}"; then
            tail -n +2 "${QUEUE_FILE}" >"${QUEUE_FILE}.tmp"
            mv -- "${QUEUE_FILE}.tmp" "${QUEUE_FILE}"
            printf '%s' "${job}"
        fi
    ) 9>"${STATUS_DIR}/queue.lock"
}

run_worker() {
    local worker=$1 gpu=${GPU_IDS[$(($1 / WORKERS_PER_GPU))]} eval_pid=''
    local job config task output log status_file code rate status
    # The evaluator has its own process group, including renderer/tee subprocesses.
    trap 'if [[ -n ${eval_pid} ]]; then kill -TERM -- "-${eval_pid}" 2>/dev/null || true; wait "${eval_pid}" || true; fi; exit 143' INT TERM
    while true; do
        job=$(claim_next_job)
        [[ -n ${job} ]] || break
        config=${job%%:*}; task=${job#*:}
        output=${RUN_ROOT}/${config}/${task}/${ATTEMPT}
        log=${LOG_DIR}/${config}_${task}_${ATTEMPT}_worker${worker}.log
        status_file=${STATUS_DIR}/${config}_${task}.tsv
        printf '[GPU %s worker %s] START %s/%s | log=%s\n' "${gpu}" "${worker}" "${config}" "${task}" "${log}"
        # The parent captures both streams; disable the single-task launcher's second log copy.
        CUDA_VISIBLE_DEVICES=${gpu} TASK_NAME=${task} TASK_CONFIG=${config} OUTPUT_DIR=${output} LOG_FILE=/dev/null \
            setsid timeout --kill-after=10s "${TASK_TIMEOUT}" bash "${SCRIPT_DIR}/eval_robotwin.sh" >"${log}" 2>&1 &
        eval_pid=$!
        code=0
        wait "${eval_pid}" || code=$?
        eval_pid=''
        status=FAILED; rate='-'
        if ((code == 0)); then
            if rate=$(result_rate "${output}" "${task}" 2>>"${log}"); then status=OK; else code=1; rate='-'; fi
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${config}" "${task}" "${gpu}" "${worker}" \
            "${status}" "${code}" "${rate}" "${output}" "${log}" >"${status_file}.tmp"
        mv -- "${status_file}.tmp" "${status_file}"
        if [[ ${status} == OK ]]; then
            printf '[GPU %s worker %s] DONE  %s/%s | success=%.2f%% | episodes=%s | log=%s\n' "${gpu}" "${worker}" "${config}" "${task}" "${rate}" "${N_EPISODES}" "${log}"
        else
            printf '[GPU %s worker %s] FAILED %s/%s | exit=%s | log=%s\n' "${gpu}" "${worker}" "${config}" "${task}" "${code}" "${log}"
        fi
        summarize
    done
}

echo "Run: ${RUN_ROOT} | GPUs: ${GPU_IDS[*]} | workers/GPU: ${WORKERS_PER_GPU} | episodes/task: ${N_EPISODES}"
echo "Jobs: ${#JOBS[@]} | pending: ${#PENDING[@]} | completed skipped: $((${#JOBS[@]} - ${#PENDING[@]}))"
echo 'LIVE scores cover completed tasks only; incomplete runs are not full benchmark scores.'
summarize
PIDS=()
cleanup() {
    trap - INT TERM
    for pid in "${PIDS[@]}"; do kill -TERM "${pid}" 2>/dev/null || true; done
    for pid in "${PIDS[@]}"; do wait "${pid}" || true; done
    summarize || true
    exit 143
}
trap cleanup INT TERM
workers_to_start=$((NUM_GPUS * WORKERS_PER_GPU))
((workers_to_start <= ${#PENDING[@]})) || workers_to_start=${#PENDING[@]}
for ((worker=0; worker<workers_to_start; worker++)); do
    run_worker "${worker}" &
    PIDS+=("$!")
    sleep "${STARTUP_STAGGER_SECONDS}"
done
worker_failure=0
for pid in "${PIDS[@]}"; do wait "${pid}" || worker_failure=1; done
summarize final
exit "${worker_failure}"
