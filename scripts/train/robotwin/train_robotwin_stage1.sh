#!/usr/bin/env bash
# WAVE-0 Stage 1 on RoboTwin: single-machine training with one or more GPUs.
# Activate the installed WAVE-0 Python environment before running this script.
set -euo pipefail

# Paths
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
DATASET_ROOT=${DATASET_ROOT:-/path/to/datasets/robotwin_unified}
PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH:-/path/to/models/pi05_base}
DATASET_STATS_PATH=${DATASET_STATS_PATH:-${DATASET_ROOT}/meta/stats.json}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/outputs/train/robotwin}

# Training
NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-64}
STEPS=${STEPS:-50000}
SAVE_FREQ=${SAVE_FREQ:-5000}
LOG_FREQ=${LOG_FREQ:-200}
LR=${LR:-1e-4}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
FINAL_LR=${FINAL_LR:-2.5e-6}
FUSED_ADAMW=${FUSED_ADAMW:-true}
SEED=${SEED:-42}

# Data loading and compilation
NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-true}
VIDEO_DECODE_DTYPE=${VIDEO_DECODE_DTYPE:-uint8}
COMPILE_MODEL=${COMPILE_MODEL:-true}
COMPILE_MODE=${COMPILE_MODE:-default}
COMPUTE_ZERO_ACTION_METRIC=${COMPUTE_ZERO_ACTION_METRIC:-false}
if (( NUM_WORKERS == 0 )); then
    PERSISTENT_WORKERS=false
fi

# Outputs and logging (keep logs outside the experiment output directory)
EXPERIMENT_NAME=${EXPERIMENT_NAME:-wave0_robotwin_stage1_bs$((NUM_GPUS * BATCH_SIZE))_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=$(realpath -m -- "${OUTPUT_DIR:-${OUTPUT_ROOT}/${EXPERIMENT_NAME}}")
LOG_ROOT=${LOG_ROOT:-${OUTPUT_ROOT}/logs}
LOG_FILE=$(realpath -m -- "${LOG_ROOT}/${EXPERIMENT_NAME}.log")
WANDB_ENABLE=${WANDB_ENABLE:-false}
WANDB_PROJECT=${WANDB_PROJECT:-wave0}
case "${LOG_FILE}" in
    "${OUTPUT_DIR}"/*) echo "LOG_ROOT must be outside OUTPUT_DIR." >&2; exit 2 ;;
esac

# Runtime: use the active environment, without machine-specific cache/library paths.
PYTHON=${PYTHON:-python}
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}

MULTI_GPU_ARGS=()
if (( NUM_GPUS > 1 )); then
    MULTI_GPU_ARGS=(--multi_gpu)
fi

mkdir -p "${LOG_ROOT}"
{
    echo "Started: $(date -u +%FT%TZ)"
    echo "Experiment: ${EXPERIMENT_NAME}"
    echo "Output: ${OUTPUT_DIR}"
    echo "Training log: ${LOG_FILE}"
    echo "Base PI0.5: ${PRETRAINED_MODEL_PATH}"
    echo "Dataset: ${DATASET_ROOT}"
    echo "Normalization statistics: ${DATASET_STATS_PATH}"
    echo "GPUs / per-GPU batch / global batch: ${NUM_GPUS} / ${BATCH_SIZE} / $((NUM_GPUS * BATCH_SIZE))"
    echo "Steps / save frequency: ${STEPS} / ${SAVE_FREQ}"
    echo "LR / warmup / final LR: ${LR} / ${WARMUP_STEPS} / ${FINAL_LR}"
    echo "Data workers / prefetch / persistent: ${NUM_WORKERS} / ${PREFETCH_FACTOR} / ${PERSISTENT_WORKERS}"
    echo "Video decode dtype / fused AdamW: ${VIDEO_DECODE_DTYPE} / ${FUSED_ADAMW}"
    echo "Frozen: PI0.5 vision target backbone"
    echo "Trainable: world action encoder + world state encoder + latent world model"

    "${PYTHON}" -m accelerate.commands.launch \
        "${MULTI_GPU_ARGS[@]}" \
        --num_processes="${NUM_GPUS}" \
        --num_machines=1 \
        --machine_rank=0 \
        --dynamo_backend=no \
        --mixed_precision=bf16 \
        -m lerobot.scripts.lerobot_train \
        --dataset.repo_id=lerobot/robotwin_unified \
        --dataset.root="${DATASET_ROOT}" \
        --dataset.stats_path="${DATASET_STATS_PATH}" \
        --dataset.video_backend=torchcodec \
        --dataset.video_decode_dtype="${VIDEO_DECODE_DTYPE}" \
        --dataset.image_transforms.enable=false \
        --dataset.use_imagenet_stats=false \
        --policy.type=wave0 \
        --policy.pretrained_path="${PRETRAINED_MODEL_PATH}" \
        --policy.world_model_base_model_path="${PRETRAINED_MODEL_PATH}" \
        --policy.world_model_stage=stage1 \
        --policy.world_model_future_horizons='[31]' \
        --policy.world_model_width=1024 \
        --policy.world_model_depth=12 \
        --policy.world_model_num_heads=16 \
        --policy.world_model_mlp_dim=4096 \
        --policy.world_action_encoder_depth=2 \
        --policy.world_model_spatial_pool_size=4 \
        --policy.world_model_future_loss_weight=1.0 \
        --policy.world_model_delta_loss_weight=0.5 \
        --policy.world_model_rank_loss_weight=0.1 \
        --policy.world_model_rank_margin=0.1 \
        --policy.world_model_compute_zero_action_metric="${COMPUTE_ZERO_ACTION_METRIC}" \
        --policy.chunk_size=32 \
        --policy.n_action_steps=32 \
        --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
        --policy.compile_model="${COMPILE_MODEL}" \
        --policy.compile_mode="${COMPILE_MODE}" \
        --policy.gradient_checkpointing=false \
        --policy.dtype=bfloat16 \
        --policy.device=cuda \
        --policy.push_to_hub=false \
        --policy.world_model_optimizer_lr="${LR}" \
        --policy.optimizer_betas='[0.9,0.95]' \
        --policy.optimizer_eps=1e-8 \
        --policy.optimizer_weight_decay=0.01 \
        --policy.optimizer_grad_clip_norm=1.0 \
        --policy.optimizer_fused="${FUSED_ADAMW}" \
        --policy.scheduler_warmup_steps="${WARMUP_STEPS}" \
        --policy.scheduler_decay_steps="${STEPS}" \
        --policy.scheduler_decay_lr="${FINAL_LR}" \
        --output_dir="${OUTPUT_DIR}" \
        --job_name="${EXPERIMENT_NAME}" \
        --steps="${STEPS}" \
        --save_freq="${SAVE_FREQ}" \
        --log_freq="${LOG_FREQ}" \
        --eval_freq=0 \
        --batch_size="${BATCH_SIZE}" \
        --num_workers="${NUM_WORKERS}" \
        --dataloader_prefetch_factor="${PREFETCH_FACTOR}" \
        --dataloader_persistent_workers="${PERSISTENT_WORKERS}" \
        --seed="${SEED}" \
        --wandb.enable="${WANDB_ENABLE}" \
        --wandb.project="${WANDB_PROJECT}" \
        "$@"
} 2>&1 | tee -a "${LOG_FILE}"
