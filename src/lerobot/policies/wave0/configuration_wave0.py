#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

DEFAULT_IMAGE_SIZE = 224
WORLD_MODEL_FUTURE_IMAGE_PREFIX = "observation.world_model.future"


def world_model_future_image_key(image_key: str, horizon: int) -> str:
    """Return the internal observation key used for a future camera frame."""
    observation_prefix = "observation."
    suffix = image_key.removeprefix(observation_prefix)
    return f"{WORLD_MODEL_FUTURE_IMAGE_PREFIX}.h{horizon}.{suffix}"


@PreTrainedConfig.register_subclass("wave0")
@dataclass
class WAVE0Config(PreTrainedConfig):
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    image_resolution: tuple[int, int] = (
        DEFAULT_IMAGE_SIZE,
        DEFAULT_IMAGE_SIZE,
    )  # see openpi `preprocessing_pytorch.py`

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    tokenizer_max_length: int = 200  # see openpi `__post_init__`

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for state
            "ACTION": NormalizationMode.QUANTILES,  # Pi0.5 uses quantiles for action
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Finetuning settings
    freeze_vision_encoder: bool = False  # Freeze only the vision encoder
    train_expert_only: bool = False  # Freeze entire VLM, train only action expert and projections

    # Latent World Model settings. Stage 1 loads only the frozen PI0.5 vision
    # backbone and trains the modules whose names start with ``world_``.
    world_model_stage: str = "stage1"
    world_model_future_horizons: tuple[int, ...] = (31,)
    # P-large preset aligned with LightWAM. WAVE-0 additionally keeps the full
    # action-token sequence while using its pooled representation for AdaLN-Zero.
    world_model_width: int = 1024
    world_model_depth: int = 12
    world_model_num_heads: int = 16
    world_model_mlp_dim: int = 4096
    world_action_encoder_depth: int = 2
    world_model_spatial_pool_size: int = 4
    world_model_future_loss_weight: float = 1.0
    world_model_delta_loss_weight: float = 0.5
    world_model_rank_loss_weight: float = 0.1
    world_model_rank_margin: float = 0.1
    world_model_compute_zero_action_metric: bool = True
    world_model_optimizer_lr: float = 1e-4
    # Stage 2 starts from a complete PI0.5 policy and injects/freeze-loads the
    # three Stage-1 world modules from this compact checkpoint.
    world_model_stage1_checkpoint_path: str | None = None
    # Permanent reference to the PI0.5 checkpoint that supplies the frozen vision
    # tower. Unlike ``pretrained_path``, this is not replaced by a Stage-1 resume
    # checkpoint path.
    world_model_base_model_path: str | None = None

    # Pre-expert WAM settings. WAVE-0 always follows the
    # VLM -> World Model -> Action Expert path.
    # One query per step covered by the Stage-1 future horizon: h=31 -> 32 tokens.
    wam_latent_action_num_queries: int = 32
    # Resample each camera's 4x4 predicted-future grid to 2x2 before the Expert.
    wam_future_condition_spatial_pool_size: int = 2
    wam_detach_future_condition: bool = True
    # When the policy vision path is frozen, reuse it as the World Model target
    # encoder instead of constructing and checkpointing a duplicate copy.
    wam_share_frozen_vision_encoder: bool = False

    # The frozen Stage-1 action encoder teaches the VLM latent-action branch the
    # input manifold expected by the frozen World Model.
    wam_alignment_cosine_weight: float = 1.0
    wam_alignment_huber_weight: float = 0.5
    wam_alignment_initial_weight: float = 1.0
    wam_alignment_final_weight: float = 0.1
    wam_alignment_decay_start_step: int = 5_000
    wam_alignment_decay_end_step: int = 20_000
    # Weight of the complete pre-expert WAM objective relative to PI0.5 flow
    # matching. This objective contains future/delta prediction and the scheduled
    # latent-action teacher alignment term.
    wam_loss_weight: float = 0.1
    wam_loss_warmup_steps: int = 1_000
    # Optional diagnostic only. It adds a second frozen World Model pass and is
    # therefore disabled by default for full-scale training.
    wam_compute_shuffled_action_metric: bool = False

    # Optimizer settings: see openpi `AdamW`
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    optimizer_fused: bool = False

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.world_model_stage not in {"stage1", "stage2", "inference", "disabled"}:
            raise ValueError(
                "world_model_stage must be one of 'stage1', 'stage2', 'inference', "
                "or 'disabled'; "
                f"got {self.world_model_stage!r}."
            )
        if self.world_model_stage == "stage2" and not self.world_model_stage1_checkpoint_path:
            raise ValueError("world_model_stage1_checkpoint_path is required for Stage-2 training.")
        if self.wam_share_frozen_vision_encoder and not self.freeze_vision_encoder:
            raise ValueError("wam_share_frozen_vision_encoder requires freeze_vision_encoder=True.")
        if len(self.world_model_future_horizons) != 1:
            raise ValueError(
                "The first WAVE-0 implementation supports exactly one future horizon; "
                f"got {self.world_model_future_horizons}."
            )
        horizon = self.world_model_future_horizons[0]
        if horizon < 1 or horizon >= self.chunk_size:
            raise ValueError(
                "The future horizon must be in [1, chunk_size - 1]; "
                f"got horizon={horizon}, chunk_size={self.chunk_size}."
            )
        if self.world_model_width % self.world_model_num_heads != 0:
            raise ValueError(
                "world_model_width must be divisible by world_model_num_heads; "
                f"got {self.world_model_width} and {self.world_model_num_heads}."
            )
        if self.world_model_spatial_pool_size <= 0:
            raise ValueError("world_model_spatial_pool_size must be positive.")
        if self.wam_latent_action_num_queries <= 0:
            raise ValueError("wam_latent_action_num_queries must be positive.")
        if self.wam_latent_action_num_queries != horizon + 1:
            raise ValueError(
                "The first pre-expert WAM implementation requires one latent-action query "
                "per action from t through the future horizon; expected "
                f"{horizon + 1}, got {self.wam_latent_action_num_queries}."
            )
        if not (0 < self.wam_future_condition_spatial_pool_size <= self.world_model_spatial_pool_size):
            raise ValueError(
                "wam_future_condition_spatial_pool_size must be in "
                f"[1, {self.world_model_spatial_pool_size}]."
            )
        if self.wam_alignment_cosine_weight < 0:
            raise ValueError("wam_alignment_cosine_weight must be non-negative.")
        if self.wam_alignment_huber_weight < 0:
            raise ValueError("wam_alignment_huber_weight must be non-negative.")
        if self.wam_alignment_initial_weight < self.wam_alignment_final_weight:
            raise ValueError(
                "wam_alignment_initial_weight must be greater than or equal to wam_alignment_final_weight."
            )
        if self.wam_alignment_final_weight < 0:
            raise ValueError("wam_alignment_final_weight must be non-negative.")
        if self.wam_alignment_decay_start_step < 0:
            raise ValueError("wam_alignment_decay_start_step must be non-negative.")
        if self.wam_alignment_decay_end_step <= self.wam_alignment_decay_start_step:
            raise ValueError(
                "wam_alignment_decay_end_step must be greater than wam_alignment_decay_start_step."
            )
        if self.wam_loss_weight < 0:
            raise ValueError("wam_loss_weight must be non-negative.")
        if self.wam_loss_warmup_steps < 0:
            raise ValueError("wam_loss_warmup_steps must be non-negative.")

    def get_wam_alignment_weight(self, step: int) -> float:
        """Return the persistent latent-action teacher weight at a training step.

        The first implementation keeps weight 1.0 through step 5k, applies a
        smooth cosine decay to 0.1 from 5k to 20k, and then keeps the 0.1 floor
        for the rest of the 100k run.
        """
        if step < 0:
            raise ValueError(f"step must be non-negative, got {step}.")
        if step <= self.wam_alignment_decay_start_step:
            return self.wam_alignment_initial_weight
        if step >= self.wam_alignment_decay_end_step:
            return self.wam_alignment_final_weight

        decay_span = self.wam_alignment_decay_end_step - self.wam_alignment_decay_start_step
        progress = (step - self.wam_alignment_decay_start_step) / decay_span
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (
            self.wam_alignment_final_weight
            + (self.wam_alignment_initial_weight - self.wam_alignment_final_weight) * cosine_factor
        )

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = OBS_IMAGES + f".empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if OBS_STATE not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features[OBS_STATE] = state_feature

        if ACTION not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features[ACTION] = action_feature

    @property
    def world_model_image_features(self) -> dict[str, PolicyFeature]:
        """Return only real cameras used as World Model inputs and targets.

        ``empty_cameras`` adds masked visual placeholders so the PI0.5 VLM can
        preserve the camera-slot layout used during pretraining. Those placeholders
        contain no observations or future frames and therefore must not be treated
        as physical cameras by the latent World Model.
        """
        empty_camera_prefix = f"{OBS_IMAGES}.empty_camera_"
        return {
            key: feature
            for key, feature in self.image_features.items()
            if not key.startswith(empty_camera_prefix)
        }

    def get_optimizer_preset(self) -> AdamWConfig:
        lr = self.world_model_optimizer_lr if self.world_model_stage == "stage1" else self.optimizer_lr
        return AdamWConfig(
            lr=lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
            fused=self.optimizer_fused,
        )

    def get_scheduler_preset(self):
        peak_lr = self.world_model_optimizer_lr if self.world_model_stage == "stage1" else self.optimizer_lr
        decay_lr = min(self.scheduler_decay_lr, peak_lr)
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=peak_lr,
            decay_lr=decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def observation_delta_indices_by_key(self) -> dict[str, list[int]]:
        """Load current and future frames for cameras without time-stacking robot state.

        The wildcard is resolved by ``datasets.factory.resolve_delta_timestamps``.
        It deliberately applies only to ``observation.images.*`` features.
        """
        if self.world_model_stage not in {"stage1", "stage2"}:
            return {}
        return {
            f"{OBS_IMAGES}.*": [0, *self.world_model_future_horizons],
        }

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
