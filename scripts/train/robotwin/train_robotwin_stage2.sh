#!/usr/bin/env bash
# WAVE-0 Stage 2 on RoboTwin: single-machine training with one or more GPUs.
# Activate the installed WAVE-0 Python environment before running this script.
set -euo pipefail

# Paths: initialize the policy from official PI0.5, not a fine-tuned policy.
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
DATASET_ROOT=${DATASET_ROOT:-/path/to/datasets/robotwin_unified}
DATASET_STATS_PATH=${DATASET_STATS_PATH:-${DATASET_ROOT}/meta/stats.json}
POLICY_BASE_PATH=${POLICY_BASE_PATH:-/path/to/models/pi05_base}
STAGE1_CHECKPOINT_PATH=${STAGE1_CHECKPOINT_PATH:-/path/to/stage1/checkpoints/050000/pretrained_model}
# The compact Stage-1 checkpoint does not contain its frozen vision backbone.
WORLD_VISION_BASE_PATH=${WORLD_VISION_BASE_PATH:-${POLICY_BASE_PATH}}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/outputs/train/robotwin}

# Training: 8 GPUs x 64 examples = global batch 512.
NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-64}
STEPS=${STEPS:-100000}
SAVE_FREQ=${SAVE_FREQ:-10000}
LOG_FREQ=${LOG_FREQ:-200}
LR=${LR:-1e-4}
WARMUP_STEPS=${WARMUP_STEPS:-1000}
FINAL_LR=${FINAL_LR:-1e-6}
FUSED_ADAMW=${FUSED_ADAMW:-true}
SEED=${SEED:-42}

# Pre-expert world-action supervision (use the validated lambda=1.0 recipe).
# Stage 2 uses future + 0.5*delta; the Stage-1 rank loss is not used here.
WAM_LOSS_WEIGHT=${WAM_LOSS_WEIGHT:-1.0}
WAM_LOSS_WARMUP_STEPS=${WAM_LOSS_WARMUP_STEPS:-1000}
ALIGN_INITIAL_WEIGHT=${ALIGN_INITIAL_WEIGHT:-1.0}
ALIGN_FINAL_WEIGHT=${ALIGN_FINAL_WEIGHT:-0.1}
ALIGN_DECAY_START=${ALIGN_DECAY_START:-5000}
ALIGN_DECAY_END=${ALIGN_DECAY_END:-20000}
SHUFFLED_ACTION_METRIC=${SHUFFLED_ACTION_METRIC:-false}

# Data loading and compilation
NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-true}
VIDEO_DECODE_DTYPE=${VIDEO_DECODE_DTYPE:-uint8}
COMPILE_MODEL=${COMPILE_MODEL:-true}
COMPILE_MODE=${COMPILE_MODE:-default}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}
if (( NUM_WORKERS == 0 )); then
    PERSISTENT_WORKERS=false
fi

# Outputs and logging (keep logs outside the experiment output directory)
EXPERIMENT_NAME=${EXPERIMENT_NAME:-wave0_robotwin_stage2_lam${WAM_LOSS_WEIGHT}_bs$((NUM_GPUS * BATCH_SIZE))_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=$(realpath -m -- "${OUTPUT_DIR:-${OUTPUT_ROOT}/${EXPERIMENT_NAME}}")
LOG_ROOT=${LOG_ROOT:-${OUTPUT_ROOT}/logs}
LOG_FILE=$(realpath -m -- "${LOG_ROOT}/${EXPERIMENT_NAME}.log")
WANDB_ENABLE=${WANDB_ENABLE:-false}
WANDB_PROJECT=${WANDB_PROJECT:-wave0}
case "${LOG_FILE}" in
    "${OUTPUT_DIR}"|"${OUTPUT_DIR}"/*) echo "LOG_ROOT must be outside OUTPUT_DIR." >&2; exit 2 ;;
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
    echo "Policy initialization: ${POLICY_BASE_PATH}"
    echo "Frozen Stage-1 world modules: ${STAGE1_CHECKPOINT_PATH}"
    echo "Frozen current/future target vision: ${WORLD_VISION_BASE_PATH}"
    echo "Dataset: ${DATASET_ROOT}"
    echo "Normalization statistics: ${DATASET_STATS_PATH}"
    echo "GPUs / per-GPU batch / global batch: ${NUM_GPUS} / ${BATCH_SIZE} / $((NUM_GPUS * BATCH_SIZE))"
    echo "Steps / save frequency: ${STEPS} / ${SAVE_FREQ}"
    echo "LR / warmup / final LR: ${LR} / ${WARMUP_STEPS} / ${FINAL_LR}"
    echo "Loss: flow + ${WAM_LOSS_WEIGHT} * (future + 0.5*delta + lambda_align*(cosine + 0.5*Huber))"
    echo "WAM loss warmup: ${WAM_LOSS_WARMUP_STEPS}"
    echo "Alignment: ${ALIGN_INITIAL_WEIGHT} -> ${ALIGN_FINAL_WEIGHT}, decay ${ALIGN_DECAY_START}-${ALIGN_DECAY_END}"
    echo "Future condition: 3 cameras x 2x2 tokens; detach predicted future before resampling"
    echo "Data workers / prefetch / persistent: ${NUM_WORKERS} / ${PREFETCH_FACTOR} / ${PERSISTENT_WORKERS}"
    echo "Video decode dtype / fused AdamW: ${VIDEO_DECODE_DTYPE} / ${FUSED_ADAMW}"
    echo "torch.compile / mode / gradient checkpointing: ${COMPILE_MODEL} / ${COMPILE_MODE} / ${GRADIENT_CHECKPOINTING}"
    echo "Trainable: VLM including policy vision, latent queries/projection, future resampler, Action Expert/action projections"
    echo "Frozen: world action encoder (GT-action teacher), world state encoder, latent world model, target vision backbone"

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
        --policy.pretrained_path="${POLICY_BASE_PATH}" \
        --policy.world_model_stage=stage2 \
        --policy.world_model_stage1_checkpoint_path="${STAGE1_CHECKPOINT_PATH}" \
        --policy.world_model_base_model_path="${WORLD_VISION_BASE_PATH}" \
        --policy.world_model_future_horizons='[31]' \
        --policy.world_model_width=1024 \
        --policy.world_model_depth=12 \
        --policy.world_model_num_heads=16 \
        --policy.world_model_mlp_dim=4096 \
        --policy.world_action_encoder_depth=2 \
        --policy.world_model_spatial_pool_size=4 \
        --policy.world_model_future_loss_weight=1.0 \
        --policy.world_model_delta_loss_weight=0.5 \
        --policy.world_model_compute_zero_action_metric=false \
        --policy.wam_latent_action_num_queries=32 \
        --policy.wam_future_condition_spatial_pool_size=2 \
        --policy.wam_detach_future_condition=true \
        --policy.wam_share_frozen_vision_encoder=false \
        --policy.wam_alignment_cosine_weight=1.0 \
        --policy.wam_alignment_huber_weight=0.5 \
        --policy.wam_alignment_initial_weight="${ALIGN_INITIAL_WEIGHT}" \
        --policy.wam_alignment_final_weight="${ALIGN_FINAL_WEIGHT}" \
        --policy.wam_alignment_decay_start_step="${ALIGN_DECAY_START}" \
        --policy.wam_alignment_decay_end_step="${ALIGN_DECAY_END}" \
        --policy.wam_loss_weight="${WAM_LOSS_WEIGHT}" \
        --policy.wam_loss_warmup_steps="${WAM_LOSS_WARMUP_STEPS}" \
        --policy.wam_compute_shuffled_action_metric="${SHUFFLED_ACTION_METRIC}" \
        --policy.chunk_size=32 \
        --policy.n_action_steps=32 \
        --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
        --policy.compile_model="${COMPILE_MODEL}" \
        --policy.compile_mode="${COMPILE_MODE}" \
        --policy.gradient_checkpointing="${GRADIENT_CHECKPOINTING}" \
        --policy.dtype=bfloat16 \
        --policy.freeze_vision_encoder=false \
        --policy.train_expert_only=false \
        --policy.device=cuda \
        --policy.push_to_hub=false \
        --policy.optimizer_lr="${LR}" \
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
