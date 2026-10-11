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

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.wave0.configuration_wave0 import (
    WAVE0Config,
    world_model_future_image_key,
)
from lerobot.policies.wave0.modeling_wave0 import pad_vector
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


@ProcessorStepRegistry.register(name="wave0_prepare_state_tokenizer_processor_step")
@dataclass
class Wave0PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"

    def get_config(self) -> dict[str, Any]:
        """Return all constructor arguments needed to restore this step."""
        return {
            "max_state_dim": self.max_state_dim,
            "task_key": self.task_key,
        }

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # Keep the unmodified task labels for same-task action shuffling in the
        # Stage-1 action-sensitivity loss. The tokenizer replaces ``task`` below.
        transition[TransitionKey.COMPLEMENTARY_DATA]["world_model_task"] = list(tasks)

        # TODO: check if this necessary
        state = deepcopy(state)

        # Prepare state (pad to max_state_dim)
        state = pad_vector(state, self.max_state_dim)

        # State should already be normalized to [-1, 1] by the NormalizerProcessorStep that runs before this step
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        state_np = state.cpu().numpy()
        discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        full_prompts = []
        for i, task in enumerate(tasks):
            cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
            state_str = " ".join(map(str, discretized_states[i]))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            full_prompts.append(full_prompt)

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


@ProcessorStepRegistry.register(name="wave0_scale_uint8_images_processor_step")
@dataclass
class Wave0ScaleUint8ImagesProcessorStep(ProcessorStep):
    """Scale raw uint8 camera frames to float32 in [0, 1].

    This is the exact conversion the DataLoader worker used to perform in
    ``decode_video_frames`` (``frames / 255``). Deferring it until after
    ``DeviceProcessorStep`` means the worker handoff, page pinning, and the
    host-to-device copy all carry uint8 instead of float32, which is 4x less
    traffic, and the divide itself runs on the accelerator.

    The step is a no-op for float inputs, so pipelines fed by a ``float32``
    dataset and environment inference are unaffected.

    The divisor is a tensor rather than the Python scalar ``255.0`` on purpose.
    PyTorch lowers a scalar division to a reciprocal multiply on CUDA, which is
    not correctly rounded and disagrees with the CPU divide it replaces on 126
    of the 256 byte values by one float32 ULP. A tensor divisor uses the
    correctly rounded kernel, so the output is bit-identical to the old
    worker-side ``frames / 255``.
    """

    image_key_prefix: str = OBS_IMAGES
    _divisors: dict[torch.device, torch.Tensor] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def get_config(self) -> dict[str, Any]:
        """Return all constructor arguments needed to restore this step."""
        return {"image_key_prefix": self.image_key_prefix}

    def _divisor(self, device: torch.device) -> torch.Tensor:
        divisor = self._divisors.get(device)
        if divisor is None:
            divisor = torch.tensor(255.0, dtype=torch.float32, device=device)
            self._divisors[device] = divisor
        return divisor

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {}) or {}
        converted = {
            key: value.to(dtype=torch.float32).div_(self._divisor(value.device))
            for key, value in observation.items()
            if key.startswith(self.image_key_prefix)
            and isinstance(value, torch.Tensor)
            and value.dtype == torch.uint8
        }
        if not converted:
            return transition

        transition = transition.copy()
        transition[TransitionKey.OBSERVATION] = {**observation, **converted}
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="wave0_split_future_images_processor_step")
@dataclass
class Wave0SplitFutureImagesProcessorStep(ProcessorStep):
    """Split dataset time-stacked camera tensors into current and future observations.

    LeRobot returns a camera requested at ``[0, h]`` as ``[B, 2, C, H, W]``.
    The original camera key is kept for the current frame so the standard PI0.5
    inference path remains unchanged. Future frames receive internal observation
    keys that are consumed only by the Stage-1 world-model forward.
    """

    # The defaults keep checkpoints saved before this step implemented
    # ``get_config`` loadable. Those checkpoints stored an empty config. A
    # no-op split is the correct behaviour for policy evaluation, where the
    # environment supplies only the current frame rather than time-stacked
    # current/future images.
    image_keys: tuple[str, ...] = ()
    future_horizons: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        # JSON serialization restores tuples as lists.
        self.image_keys = tuple(self.image_keys)
        self.future_horizons = tuple(int(horizon) for horizon in self.future_horizons)

    def get_config(self) -> dict[str, Any]:
        """Return all constructor arguments needed to restore this step."""
        return {
            "image_keys": list(self.image_keys),
            "future_horizons": list(self.future_horizons),
        }

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION, {}) or {})
        expected_time_steps = 1 + len(self.future_horizons)

        for image_key in self.image_keys:
            image = observation.get(image_key)
            if image is None or not isinstance(image, torch.Tensor):
                continue
            if image.ndim == 4:
                # Normal environment inference provides only the current frame.
                continue
            if image.ndim != 5:
                raise ValueError(
                    f"Expected camera {image_key!r} to have shape [B,C,H,W] or [B,T,C,H,W], "
                    f"got {tuple(image.shape)}."
                )
            if image.shape[1] != expected_time_steps:
                raise ValueError(
                    f"Camera {image_key!r} has {image.shape[1]} time steps, but WAVE-0 expected "
                    f"{expected_time_steps} for indices [0, {', '.join(map(str, self.future_horizons))}]."
                )

            observation[image_key] = image[:, 0]
            for time_index, horizon in enumerate(self.future_horizons, start=1):
                observation[world_model_future_image_key(image_key, horizon)] = image[:, time_index]

        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_wave0_pre_post_processors(
    config: WAVE0Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        # NOTE: NormalizerProcessorStep MUST come before Wave0PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
    ]

    # Stage 1 never calls the language model. Avoid loading or running the
    # PaliGemma tokenizer; raw task labels remain available for negative-action
    # construction. Non-Stage-1 modes preserve the original PI0.5 pipeline.
    if config.world_model_stage != "stage1":
        input_steps.extend(
            [
                Wave0PrepareStateTokenizerProcessorStep(max_state_dim=config.max_state_dim),
                TokenizerProcessorStep(
                    tokenizer_name="google/paligemma-3b-pt-224",
                    max_length=config.tokenizer_max_length,
                    padding_side="right",
                    padding="max_length",
                ),
            ]
        )
    input_steps.append(DeviceProcessorStep(device=config.device))
    # Raw uint8 frames (``--dataset.video_decode_dtype=uint8``) are scaled on device,
    # after the transfer, so the PCIe copy carries a quarter of the bytes.
    input_steps.append(Wave0ScaleUint8ImagesProcessorStep())
    # The split MUST stay after the device step. Splitting on the host produces one
    # non-contiguous view per timestamp, and a non-contiguous CPU tensor cannot use
    # the pinned async copy path, so each camera/horizon became its own blocking
    # host-to-device transfer. Keeping the time-stacked camera tensor whole lets
    # DataLoader pinning work and turns 1 + len(horizons) transfers per camera into
    # a single asynchronous one; the slicing itself is then a free device view.
    input_steps.append(
        Wave0SplitFutureImagesProcessorStep(
            # Empty PI0.5 camera slots are VLM-only masked placeholders. They do
            # not have future observations and are not World Model targets.
            image_keys=tuple(config.world_model_image_features),
            future_horizons=config.world_model_future_horizons,
        )
    )

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
