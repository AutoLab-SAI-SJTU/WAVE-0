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

import builtins
import dataclasses
import logging
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.utils.import_utils import _transformers_available

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma
    from transformers.models.gemma.modeling_gemma import GemmaForCausalLM
    from transformers.models.paligemma.modeling_paligemma import (
        PaliGemmaForConditionalGeneration,
        PaliGemmaMultiModalProjector,
    )
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel
else:
    CONFIG_MAPPING = None
    modeling_gemma = None
    GemmaForCausalLM = None
    PaliGemmaForConditionalGeneration = None
    PaliGemmaMultiModalProjector = None
    SiglipVisionModel = None

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.wave0.configuration_wave0 import (
    DEFAULT_IMAGE_SIZE,
    WAVE0Config,
    world_model_future_image_key,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    OPENPI_ATTENTION_MASK_VALUE,
)


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None
    state: Tensor | None
    world_images: list[Tensor] | None


def _is_world_model_state_key(key: str) -> bool:
    """Return whether a checkpoint tensor belongs to the WAM extension."""
    parts = key.split(".")
    while parts and parts[0] in {"model", "_orig_mod"}:
        parts.pop(0)
    if not parts:
        return False
    return parts[0] in {
        "world_action_encoder",
        "world_state_encoder",
        "world_model",
        "world_vision_backbone",
        "stage2_step",
        "wam_latent_action_queries",
        "wam_latent_action_projection",
        "wam_future_condition_resampler",
    }


def _is_wam_teacher_state_key(key: str) -> bool:
    """Return whether a tensor is needed only for Stage-2 training."""
    parts = key.split(".")
    while parts and parts[0] in {"model", "_orig_mod"}:
        parts.pop(0)
    if not parts:
        return False
    return parts[0] in {"world_action_encoder", "stage2_step"}


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def recover_clean_actions(x_t: Tensor, predicted_velocity: Tensor, time: Tensor) -> Tensor:
    """Recover the t=0 action estimate from PI0.5's flow-matching parameterization."""
    if time.ndim != 1 or time.shape[0] != x_t.shape[0]:
        raise ValueError(f"Expected time shape ({x_t.shape[0]},), got {tuple(time.shape)}.")
    if predicted_velocity.shape != x_t.shape:
        raise ValueError(
            "x_t and predicted_velocity must have identical shapes; "
            f"got {tuple(x_t.shape)} and {tuple(predicted_velocity.shape)}."
        )
    return x_t - time[:, None, None] * predicted_velocity


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(
    layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, paligemma, gemma_expert
):
    models = [paligemma.language_model, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    scaling = paligemma.language_model.layers[layer_idx].self_attn.scaling
    # Attention computation
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = paligemma.language_model.layers[layer_idx].self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
        after_first_residual = out_emb.clone()
        out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


def compute_layer_with_future_condition(
    layer_idx,
    inputs_embeds,
    attention_mask,
    position_ids,
    adarms_cond,
    paligemma,
    gemma_expert,
):
    """Run one shared-attention layer over VLM, future, and action streams.

    The future and action streams share the Action Expert weights, but use
    independent AdaRMS conditions. This retained reference implementation is
    used to verify that the cached-context Expert path is numerically equivalent
    to the original interleaved three-stream layer.
    """
    models = [
        paligemma.language_model,
        gemma_expert.model,
        gemma_expert.model,
    ]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for index, hidden_states in enumerate(inputs_embeds):
        layer = models[index].layers[layer_idx]
        hidden_states, gate = layer.input_layernorm(
            hidden_states,
            cond=adarms_cond[index],
        )
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_states.append(layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2))
        key_states.append(layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2))
        value_states.append(layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2))

    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        unsqueeze_dim=1,
    )
    scaling = paligemma.language_model.layers[layer_idx].self_attn.scaling
    attention_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    head_dim = paligemma.language_model.layers[layer_idx].self_attn.head_dim
    attention_output = attention_output.reshape(
        query_states.shape[0],
        -1,
        8 * head_dim,
    )

    outputs_embeds = []
    start_position = 0
    for index, hidden_states in enumerate(inputs_embeds):
        layer = models[index].layers[layer_idx]
        end_position = start_position + hidden_states.shape[1]
        stream_attention = attention_output[:, start_position:end_position]
        if stream_attention.dtype != layer.self_attn.o_proj.weight.dtype:
            stream_attention = stream_attention.to(layer.self_attn.o_proj.weight.dtype)
        output = layer.self_attn.o_proj(stream_attention)
        output = modeling_gemma._gated_residual(hidden_states, output, gates[index])  # noqa: SLF001
        first_residual = output
        output, gate = layer.post_attention_layernorm(
            output,
            cond=adarms_cond[index],
        )
        if output.dtype != layer.mlp.up_proj.weight.dtype:
            output = output.to(layer.mlp.up_proj.weight.dtype)
        output = layer.mlp(output)
        output = modeling_gemma._gated_residual(first_residual, output, gate)  # noqa: SLF001
        outputs_embeds.append(output)
        start_position = end_position
    return outputs_embeds


def compute_vlm_layer_with_context_kv(
    layer_idx,
    hidden_states,
    attention_mask,
    position_ids,
    context_length,
    paligemma,
):
    """Run one VLM layer and expose the original-context KV before attention.

    ``hidden_states`` contains the original PI0.5 image/language context followed
    by the internal latent-action queries.  The blockwise attention mask prevents
    context tokens from reading the queries, so the first ``context_length`` KV
    entries are numerically identical to a query-free PI0.5 prefix.  Returning
    them from every layer lets the Action Expert reuse this single differentiable
    VLM prefill instead of recomputing the VLM a second time.
    """
    layer = paligemma.language_model.layers[layer_idx]
    residual = hidden_states
    normalized, gate = layer.input_layernorm(hidden_states, cond=None)

    input_shape = normalized.shape[:-1]
    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
    query_states = layer.self_attn.q_proj(normalized).view(hidden_shape).transpose(1, 2)
    key_states = layer.self_attn.k_proj(normalized).view(hidden_shape).transpose(1, 2)
    value_states = layer.self_attn.v_proj(normalized).view(hidden_shape).transpose(1, 2)

    rotary_input = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(rotary_input, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        unsqueeze_dim=1,
    )

    # Keep the compact, non-repeated GQA KV representation
    # [B, num_kv_heads, context_length, head_dim].
    context_key = key_states[:, :, :context_length]
    context_value = value_states[:, :, :context_length]

    attention_output, _ = modeling_gemma.eager_attention_forward(
        layer.self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        layer.self_attn.scaling,
    )
    attention_output = attention_output.reshape(
        query_states.shape[0],
        -1,
        query_states.shape[1] * layer.self_attn.head_dim,
    )
    if attention_output.dtype != layer.self_attn.o_proj.weight.dtype:
        attention_output = attention_output.to(layer.self_attn.o_proj.weight.dtype)
    hidden_states = layer.self_attn.o_proj(attention_output)
    hidden_states = modeling_gemma._gated_residual(residual, hidden_states, gate)  # noqa: SLF001

    residual = hidden_states
    hidden_states, gate = layer.post_attention_layernorm(hidden_states, cond=None)
    if hidden_states.dtype != layer.mlp.up_proj.weight.dtype:
        hidden_states = hidden_states.to(layer.mlp.up_proj.weight.dtype)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = modeling_gemma._gated_residual(residual, hidden_states, gate)  # noqa: SLF001
    return hidden_states, context_key, context_value


def compute_expert_layer_with_context_kv(
    layer_idx,
    inputs_embeds,
    context_key,
    context_value,
    attention_mask,
    position_ids,
    adarms_cond,
    paligemma,
    gemma_expert,
):
    """Run future/action Expert streams against cached VLM context KV.

    Only the two Expert streams are evaluated here.  Query KV is never included:
    ``context_key`` / ``context_value`` contain exactly the original PI0.5
    image/language prefix from the matching VLM layer.
    """
    models = [gemma_expert.model, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []

    for index, hidden_states in enumerate(inputs_embeds):
        layer = models[index].layers[layer_idx]
        normalized, gate = layer.input_layernorm(
            hidden_states,
            cond=adarms_cond[index],
        )
        gates.append(gate)
        input_shape = normalized.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_states.append(layer.self_attn.q_proj(normalized).view(hidden_shape).transpose(1, 2))
        key_states.append(layer.self_attn.k_proj(normalized).view(hidden_shape).transpose(1, 2))
        value_states.append(layer.self_attn.v_proj(normalized).view(hidden_shape).transpose(1, 2))

    query_states = torch.cat(query_states, dim=2)
    local_key_states = torch.cat(key_states, dim=2)
    local_value_states = torch.cat(value_states, dim=2)
    rotary_input = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(rotary_input, position_ids)
    query_states, local_key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states,
        local_key_states,
        cos,
        sin,
        unsqueeze_dim=1,
    )

    if context_key.dtype != local_key_states.dtype:
        context_key = context_key.to(local_key_states.dtype)
    if context_value.dtype != local_value_states.dtype:
        context_value = context_value.to(local_value_states.dtype)
    key_states = torch.cat([context_key, local_key_states], dim=2)
    value_states = torch.cat([context_value, local_value_states], dim=2)

    expert_attention = gemma_expert.model.layers[layer_idx].self_attn
    attention_output, _ = modeling_gemma.eager_attention_forward(
        expert_attention,
        query_states,
        key_states,
        value_states,
        attention_mask,
        expert_attention.scaling,
    )
    attention_output = attention_output.reshape(
        query_states.shape[0],
        -1,
        query_states.shape[1] * expert_attention.head_dim,
    )

    outputs = []
    start_position = 0
    for index, hidden_states in enumerate(inputs_embeds):
        layer = models[index].layers[layer_idx]
        end_position = start_position + hidden_states.shape[1]
        stream_attention = attention_output[:, start_position:end_position]
        if stream_attention.dtype != layer.self_attn.o_proj.weight.dtype:
            stream_attention = stream_attention.to(layer.self_attn.o_proj.weight.dtype)
        output = layer.self_attn.o_proj(stream_attention)
        output = modeling_gemma._gated_residual(hidden_states, output, gates[index])  # noqa: SLF001

        residual = output
        output, gate = layer.post_attention_layernorm(
            output,
            cond=adarms_cond[index],
        )
        if output.dtype != layer.mlp.up_proj.weight.dtype:
            output = output.to(layer.mlp.up_proj.weight.dtype)
        output = layer.mlp(output)
        output = modeling_gemma._gated_residual(residual, output, gate)  # noqa: SLF001
        outputs.append(output)
        start_position = end_position
    return outputs


class GemmaConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """PaliGemma model with action expert for PI05."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = DEFAULT_IMAGE_SIZE,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
            for param in self.paligemma.vision_tower.parameters():
                param.requires_grad = False
        if self.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_model_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_model_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = suffix_model_output.past_key_values
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )

            # final norm
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values

    def forward_vlm_with_context_kv(
        self,
        *,
        prefix_embeds: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        context_length: int,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """Run one differentiable VLM prefill and retain per-layer context KV."""
        hidden_states = prefix_embeds
        context_key_values = []
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        use_gradient_checkpointing = (
            hasattr(self.paligemma.language_model, "gradient_checkpointing")
            and self.paligemma.language_model.gradient_checkpointing
            and self.training
        ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

        for layer_idx in range(num_layers):
            if use_gradient_checkpointing:
                hidden_states, context_key, context_value = torch.utils.checkpoint.checkpoint(
                    compute_vlm_layer_with_context_kv,
                    layer_idx,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    context_length,
                    use_reentrant=False,
                    preserve_rng_state=False,
                    paligemma=self.paligemma,
                )
            else:
                hidden_states, context_key, context_value = compute_vlm_layer_with_context_kv(
                    layer_idx,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    context_length,
                    paligemma=self.paligemma,
                )
            context_key_values.append((context_key, context_value))

        hidden_states, _ = self.paligemma.language_model.norm(
            hidden_states,
            cond=None,
        )
        return hidden_states, context_key_values

    def forward_expert_with_context_kv(
        self,
        *,
        context_key_values: list[tuple[Tensor, Tensor]],
        future_embeds: Tensor,
        action_embeds: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        future_adarms_cond: Tensor,
        action_adarms_cond: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run only the Expert streams against a single cached VLM prefill."""
        inputs_embeds = [future_embeds, action_embeds]
        adarms_cond = [future_adarms_cond, action_adarms_cond]
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        if len(context_key_values) != num_layers:
            raise ValueError(
                "Expected one context KV pair per transformer layer; "
                f"got {len(context_key_values)} for {num_layers} layers."
            )
        use_gradient_checkpointing = (
            hasattr(self.gemma_expert.model, "gradient_checkpointing")
            and self.gemma_expert.model.gradient_checkpointing
            and self.training
        ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

        for layer_idx, (context_key, context_value) in enumerate(context_key_values):
            if use_gradient_checkpointing:
                inputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_expert_layer_with_context_kv,
                    layer_idx,
                    inputs_embeds,
                    context_key,
                    context_value,
                    attention_mask,
                    position_ids,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                    paligemma=self.paligemma,
                    gemma_expert=self.gemma_expert,
                )
            else:
                inputs_embeds = compute_expert_layer_with_context_kv(
                    layer_idx,
                    inputs_embeds,
                    context_key,
                    context_value,
                    attention_mask,
                    position_ids,
                    adarms_cond,
                    paligemma=self.paligemma,
                    gemma_expert=self.gemma_expert,
                )

        outputs = []
        for index, hidden_states in enumerate(inputs_embeds):
            output, _ = self.gemma_expert.model.norm(
                hidden_states,
                cond=adarms_cond[index],
            )
            outputs.append(output)
        return outputs[0], outputs[1]


def spatial_pool_image_tokens(image_tokens: Tensor, output_size: int) -> Tensor:
    """Pool square SigLIP patch tokens while preserving their 2-D spatial layout."""
    if image_tokens.ndim != 3:
        raise ValueError(f"Expected image tokens with shape [B,N,D], got {tuple(image_tokens.shape)}.")
    batch_size, num_tokens, width = image_tokens.shape
    spatial_size = math.isqrt(num_tokens)
    if spatial_size * spatial_size != num_tokens:
        raise ValueError(f"Expected a square patch grid, but got {num_tokens} image tokens.")
    if output_size > spatial_size:
        raise ValueError(
            f"Cannot pool a {spatial_size}x{spatial_size} grid to the larger size {output_size}."
        )

    features = image_tokens.reshape(batch_size, spatial_size, spatial_size, width)
    features = features.permute(0, 3, 1, 2)
    features = F.adaptive_avg_pool2d(features, (output_size, output_size))
    return features.flatten(2).transpose(1, 2).contiguous()


class FrozenPaliGemmaVisionBackbone(nn.Module):
    """The only pretrained PI0.5 component required by Stage 1.

    This is equivalent to ``PaliGemmaModel.get_image_features`` but deliberately
    does not instantiate the PaliGemma language model, LM head, or action expert.
    """

    def __init__(self, config: WAVE0Config):
        super().__init__()
        vlm_config = get_gemma_config(config.paligemma_variant)
        paligemma_config = CONFIG_MAPPING["paligemma"]()
        paligemma_config.text_config.hidden_size = vlm_config.width
        paligemma_config.vision_config.image_size = config.image_resolution[0]
        paligemma_config.vision_config.intermediate_size = 4304
        paligemma_config.vision_config.projection_dim = vlm_config.width
        paligemma_config.vision_config.projector_hidden_act = "gelu_fast"
        paligemma_config.vision_config.torch_dtype = "float32"

        self.vision_tower = SiglipVisionModel(paligemma_config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(paligemma_config)
        self._set_precision(config.dtype)
        self.requires_grad_(False)
        self.eval()

    def _set_precision(self, precision: Literal["bfloat16", "float32"]) -> None:
        if precision == "float32":
            self.to(dtype=torch.float32)
            return
        if precision != "bfloat16":
            raise ValueError(f"Unsupported vision precision: {precision}")

        self.to(dtype=torch.bfloat16)
        keep_float32 = (
            "vision_model.embeddings.patch_embedding",
            "vision_model.embeddings.position_embedding",
        )
        for name, parameter in self.named_parameters():
            if any(selector in name for selector in keep_float32):
                parameter.data = parameter.data.to(dtype=torch.float32)

    def train(self, mode: bool = True):
        # The Stage-1 latent target space must remain fixed and deterministic.
        return super().train(False)

    def forward(self, image: Tensor) -> Tensor:
        vision_output = self.vision_tower(image)
        return self.multi_modal_projector(vision_output.last_hidden_state)


class SharedFrozenPaliGemmaVisionBackbone(nn.Module):
    """Unregistered view of the frozen policy vision path.

    The referenced modules remain owned by ``paligemma_with_expert``. Keeping
    the references unregistered makes the World Model consume the exact same
    tensors without adding a second set of state-dict keys to checkpoints.
    """

    def __init__(self, vision_tower: nn.Module, multi_modal_projector: nn.Module):
        super().__init__()
        object.__setattr__(self, "_vision_tower_ref", vision_tower)
        object.__setattr__(self, "_multi_modal_projector_ref", multi_modal_projector)
        self.train(False)

    def train(self, mode: bool = True):
        super().train(False)
        self._vision_tower_ref.eval()
        self._multi_modal_projector_ref.eval()
        return self

    def forward(self, image: Tensor) -> Tensor:
        vision_output = self._vision_tower_ref(image)
        return self._multi_modal_projector_ref(vision_output.last_hidden_state)


class WorldActionEncoder(nn.Module):
    """Encode a normalized GT joint-action sequence into latent action tokens."""

    def __init__(
        self,
        max_action_dim: int,
        width: int,
        depth: int,
        num_heads: int,
        max_sequence_length: int,
    ):
        super().__init__()
        hidden_width = min(width, 256)
        self.max_action_dim = max_action_dim
        self.input_projection = nn.Sequential(
            nn.Linear(2 * max_action_dim, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, width),
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, max_sequence_length, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=num_heads,
            dim_feedforward=4 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            norm=nn.LayerNorm(width),
            enable_nested_tensor=False,
        )
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, actions: Tensor, state: Tensor) -> Tensor:
        if actions.ndim != 3 or actions.shape[-1] != self.max_action_dim:
            raise ValueError(
                f"Expected padded actions [B,T,{self.max_action_dim}], got {tuple(actions.shape)}."
            )
        if state.ndim != 2 or state.shape[-1] != self.max_action_dim:
            raise ValueError(f"Expected padded state [B,{self.max_action_dim}], got {tuple(state.shape)}.")
        if actions.shape[1] > self.position_embedding.shape[1]:
            raise ValueError(
                f"Action sequence length {actions.shape[1]} exceeds configured maximum "
                f"{self.position_embedding.shape[1]}."
            )
        relative_actions = actions - state[:, None, :]
        tokens = self.input_projection(torch.cat([actions, relative_actions], dim=-1))
        tokens = tokens + self.position_embedding[:, : actions.shape[1]]
        return self.encoder(tokens)


class WorldStateEncoder(nn.Module):
    """Encode the normalized, padded robot state into one world-model token."""

    def __init__(self, max_state_dim: int, width: int):
        super().__init__()
        hidden_width = min(width, 256)
        self.max_state_dim = max_state_dim
        self.projection = nn.Sequential(
            nn.Linear(max_state_dim, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, width),
        )

    def forward(self, state: Tensor) -> Tensor:
        if state.ndim != 2 or state.shape[-1] != self.max_state_dim:
            raise ValueError(f"Expected padded state [B,{self.max_state_dim}], got {tuple(state.shape)}.")
        return self.projection(state).unsqueeze(1)


def _adaln_modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply sample-wise adaptive LayerNorm shift and scale."""
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class AdaLNWorldModelBlock(nn.Module):
    """Bidirectional transformer block with action-conditioned AdaLN-Zero."""

    def __init__(self, width: int, num_heads: int, mlp_dim: int):
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(f"width={width} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = width // num_heads

        self.attention_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.qkv_projection = nn.Linear(width, 3 * width)
        self.attention_output_projection = nn.Linear(width, width)
        self.ffn_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.ffn = nn.Sequential(
            nn.Linear(width, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, width),
        )

        # Per block: attention shift/scale/gate + FFN shift/scale/gate.
        # Zero initialization makes the new block an identity at initialization;
        # action conditioning is then learned progressively and stably.
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def _attention(self, x: Tensor) -> Tensor:
        batch_size, sequence_length, width = x.shape
        qkv = self.qkv_projection(x).reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        output = F.scaled_dot_product_attention(query, key, value)
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, width)
        return self.attention_output_projection(output)

    def forward(self, x: Tensor, action_condition: Tensor) -> Tensor:
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = self.adaln(
            action_condition
        ).chunk(6, dim=-1)
        attention_input = _adaln_modulate(self.attention_norm(x), shift_attn, scale_attn)
        x = x + gate_attn[:, None, :] * self._attention(attention_input)
        ffn_input = _adaln_modulate(self.ffn_norm(x), shift_ffn, scale_ffn)
        return x + gate_ffn[:, None, :] * self.ffn(ffn_input)


class LatentWorldModel(nn.Module):
    """Predict a future visual-latent delta from current latents, state, and actions."""

    def __init__(
        self,
        image_latent_dim: int,
        width: int,
        depth: int,
        num_heads: int,
        mlp_dim: int,
        num_cameras: int,
        spatial_tokens_per_camera: int,
    ):
        super().__init__()
        self.num_cameras = num_cameras
        self.spatial_tokens_per_camera = spatial_tokens_per_camera
        self.visual_projection = nn.Linear(image_latent_dim, width)
        self.future_output_projection = nn.Linear(width, image_latent_dim)

        self.future_queries = nn.Parameter(torch.zeros(1, num_cameras, spatial_tokens_per_camera, width))
        self.camera_embedding = nn.Parameter(torch.zeros(1, num_cameras, 1, width))
        self.horizon_embedding = nn.Parameter(torch.zeros(1, 1, width))
        self.current_type_embedding = nn.Parameter(torch.zeros(1, 1, width))
        self.state_type_embedding = nn.Parameter(torch.zeros(1, 1, width))
        self.action_type_embedding = nn.Parameter(torch.zeros(1, 1, width))
        self.query_type_embedding = nn.Parameter(torch.zeros(1, 1, width))

        self.action_condition_norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList(
            AdaLNWorldModelBlock(width=width, num_heads=num_heads, mlp_dim=mlp_dim) for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(width)

        for parameter in (
            self.future_queries,
            self.camera_embedding,
            self.horizon_embedding,
            self.current_type_embedding,
            self.state_type_embedding,
            self.action_type_embedding,
            self.query_type_embedding,
        ):
            nn.init.normal_(parameter, std=0.02)

    def forward(self, current_latent: Tensor, state_token: Tensor, action_tokens: Tensor) -> Tensor:
        expected_shape = (self.num_cameras, self.spatial_tokens_per_camera)
        if current_latent.ndim != 4 or tuple(current_latent.shape[1:3]) != expected_shape:
            raise ValueError(
                "Expected current latent [B,C,P,D] with "
                f"(C,P)={expected_shape}, got {tuple(current_latent.shape)}."
            )
        batch_size = current_latent.shape[0]

        current_tokens = self.visual_projection(current_latent)
        current_tokens = current_tokens + self.camera_embedding + self.current_type_embedding[:, None]
        current_tokens = current_tokens.flatten(1, 2)

        queries = self.future_queries + self.camera_embedding
        queries = queries.flatten(1, 2).expand(batch_size, -1, -1)
        queries = queries + self.horizon_embedding + self.query_type_embedding

        sequence = torch.cat(
            [
                current_tokens,
                state_token + self.state_type_embedding,
                action_tokens + self.action_type_embedding,
                queries,
            ],
            dim=1,
        )
        action_condition = self.action_condition_norm(action_tokens.mean(dim=1))
        output = sequence
        for block in self.blocks:
            output = block(output, action_condition)
        query_output = output[:, -queries.shape[1] :]
        query_output = self.output_norm(query_output)
        pred_delta = self.future_output_projection(query_output)
        return pred_delta.reshape(
            batch_size,
            self.num_cameras,
            self.spatial_tokens_per_camera,
            -1,
        )


class WAMFutureConditionResampler(nn.Module):
    """Convert predicted World Model latents into Action Expert condition tokens."""

    def __init__(
        self,
        *,
        input_width: int,
        output_width: int,
        num_cameras: int,
        input_spatial_size: int,
        output_spatial_size: int,
    ) -> None:
        super().__init__()
        if output_spatial_size > input_spatial_size:
            raise ValueError("Future-condition output grid cannot be larger than its input grid.")
        self.num_cameras = num_cameras
        self.input_spatial_size = input_spatial_size
        self.output_spatial_size = output_spatial_size
        self.input_norm = nn.LayerNorm(input_width)
        self.projection = nn.Linear(input_width, output_width)
        self.camera_embedding = nn.Parameter(
            torch.zeros(1, num_cameras, output_spatial_size**2, output_width)
        )
        self.type_embedding = nn.Parameter(torch.zeros(1, 1, output_width))
        nn.init.normal_(self.camera_embedding, std=0.02)
        nn.init.normal_(self.type_embedding, std=0.02)

    @property
    def num_output_tokens(self) -> int:
        return self.num_cameras * self.output_spatial_size**2

    def forward(self, predicted_future: Tensor) -> Tensor:
        expected_tokens = self.input_spatial_size**2
        if predicted_future.ndim != 4:
            raise ValueError(
                f"Predicted future latent must have shape [B,C,P,D]; got {tuple(predicted_future.shape)}."
            )
        batch_size, num_cameras, num_tokens, input_width = predicted_future.shape
        if num_cameras != self.num_cameras or num_tokens != expected_tokens:
            raise ValueError(
                "Predicted future latent has an incompatible camera/grid shape; "
                f"expected C={self.num_cameras}, P={expected_tokens}, got "
                f"C={num_cameras}, P={num_tokens}."
            )

        flattened = predicted_future.reshape(
            batch_size * num_cameras,
            num_tokens,
            input_width,
        )
        pooled = spatial_pool_image_tokens(
            flattened,
            output_size=self.output_spatial_size,
        )
        pooled = pooled.reshape(
            batch_size,
            num_cameras,
            self.output_spatial_size**2,
            input_width,
        )
        tokens = self.projection(self.input_norm(pooled))
        tokens = tokens + self.camera_embedding
        tokens = tokens.flatten(1, 2)
        return tokens + self.type_embedding


class WAVE0Stage1Model(nn.Module):
    """Minimal Stage-1 model: frozen vision backbone plus trainable world modules."""

    def __init__(self, config: WAVE0Config):
        super().__init__()
        self.config = config
        paligemma_config = get_gemma_config(config.paligemma_variant)
        num_cameras = len(config.world_model_image_features)
        if num_cameras == 0:
            raise ValueError("WAVE-0 Stage 1 requires at least one visual input feature.")

        self.vision_backbone = FrozenPaliGemmaVisionBackbone(config)
        self.world_action_encoder = WorldActionEncoder(
            max_action_dim=config.max_action_dim,
            width=config.world_model_width,
            depth=config.world_action_encoder_depth,
            num_heads=config.world_model_num_heads,
            max_sequence_length=config.chunk_size,
        )
        self.world_state_encoder = WorldStateEncoder(
            max_state_dim=config.max_state_dim,
            width=config.world_model_width,
        )
        self.world_model = LatentWorldModel(
            image_latent_dim=paligemma_config.width,
            width=config.world_model_width,
            depth=config.world_model_depth,
            num_heads=config.world_model_num_heads,
            mlp_dim=config.world_model_mlp_dim,
            num_cameras=num_cameras,
            spatial_tokens_per_camera=config.world_model_spatial_pool_size**2,
        )

        if config.compile_model:
            # Keep masking, loss assembly, and scalar diagnostics eager. Only
            # compile the fixed-shape neural modules used on every Stage-1 step.
            torch.set_float32_matmul_precision("high")
            self.vision_backbone.forward = torch.compile(
                self.vision_backbone.forward,
                mode=config.compile_mode,
            )
            self.world_action_encoder.forward = torch.compile(
                self.world_action_encoder.forward,
                mode=config.compile_mode,
            )
            self.world_state_encoder.forward = torch.compile(
                self.world_state_encoder.forward,
                mode=config.compile_mode,
            )
            self.world_model.forward = torch.compile(
                self.world_model.forward,
                mode=config.compile_mode,
            )
            logging.info(
                "Enabled module-level torch.compile for WAVE-0 Stage 1: "
                "target vision, action encoder, state encoder, and latent World Model "
                "(mode=%s)",
                config.compile_mode,
            )

    @torch.no_grad()
    def encode_world_image_pairs(
        self, current_images: list[Tensor], future_images: list[Tensor]
    ) -> tuple[Tensor, Tensor]:
        if len(current_images) != len(future_images):
            raise ValueError("Current and future image lists must contain the same cameras.")
        self.vision_backbone.eval()
        current_latents = []
        future_latents = []
        for current_image, future_image in zip(current_images, future_images, strict=True):
            batch_size = current_image.shape[0]
            image_tokens = self.vision_backbone(torch.cat([current_image, future_image], dim=0))
            pooled = spatial_pool_image_tokens(
                image_tokens,
                output_size=self.config.world_model_spatial_pool_size,
            ).to(dtype=torch.float32)
            current_latents.append(pooled[:batch_size])
            future_latents.append(pooled[batch_size:])
        return (
            torch.stack(current_latents, dim=1).detach(),
            torch.stack(future_latents, dim=1).detach(),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_backbone.eval()
        return self


class PI05Pytorch(nn.Module):  # see openpi `PI0Pytorch`
    """Core PI05 PyTorch model."""

    def __init__(self, config: WAVE0Config, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}"
            )

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # WAVE-0 obtains an explicit latent action from the VLM before calling
        # the World Model or Action Expert. Stage 1 is a compact World Model
        # trainer and deliberately does not instantiate the VLM-side branch.
        self.wam_latent_action_queries = nn.Parameter(
            torch.empty(
                1,
                config.wam_latent_action_num_queries,
                paligemma_config.width,
            )
        )
        nn.init.normal_(self.wam_latent_action_queries, std=0.02)
        self.wam_latent_action_projection = nn.Sequential(
            nn.LayerNorm(paligemma_config.width),
            nn.Linear(paligemma_config.width, config.world_model_width),
        )

        # World modules supervise Stage-2 training only. ``disabled`` is the
        # deployment/evaluation mode and deliberately constructs none of them.
        if config.world_model_stage in {"stage2", "inference"}:
            num_cameras = len(config.world_model_image_features)
            if num_cameras == 0:
                raise ValueError("WAVE-0 Stage 2/inference requires at least one visual input feature.")
            pooled_tokens = config.world_model_spatial_pool_size**2
            if config.world_model_stage == "stage2":
                self.world_action_encoder = WorldActionEncoder(
                    max_action_dim=config.max_action_dim,
                    width=config.world_model_width,
                    depth=config.world_action_encoder_depth,
                    num_heads=config.world_model_num_heads,
                    max_sequence_length=config.chunk_size,
                )
            self.world_state_encoder = WorldStateEncoder(
                max_state_dim=config.max_state_dim,
                width=config.world_model_width,
            )
            self.world_model = LatentWorldModel(
                image_latent_dim=paligemma_config.width,
                width=config.world_model_width,
                depth=config.world_model_depth,
                num_heads=config.world_model_num_heads,
                mlp_dim=config.world_model_mlp_dim,
                num_cameras=num_cameras,
                spatial_tokens_per_camera=pooled_tokens,
            )
            if config.wam_share_frozen_vision_encoder:
                # The projector is part of PaliGemma's image-feature mapping and
                # must remain fixed together with SigLIP for the Stage-1 latent
                # target space to stay unchanged.
                policy_paligemma = self.paligemma_with_expert.paligemma
                policy_paligemma.vision_tower.requires_grad_(False)
                policy_paligemma.multi_modal_projector.requires_grad_(False)
                policy_paligemma.vision_tower.eval()
                policy_paligemma.multi_modal_projector.eval()
                self.world_vision_backbone = SharedFrozenPaliGemmaVisionBackbone(
                    policy_paligemma.vision_tower,
                    policy_paligemma.multi_modal_projector,
                )
            else:
                # Keep a separate target encoder when policy vision is trainable
                # or intentionally differs from the Stage-1 latent space.
                self.world_vision_backbone = FrozenPaliGemmaVisionBackbone(config)
            self.wam_future_condition_resampler = WAMFutureConditionResampler(
                input_width=paligemma_config.width,
                output_width=action_expert_config.width,
                num_cameras=num_cameras,
                input_spatial_size=config.world_model_spatial_pool_size,
                output_spatial_size=config.wam_future_condition_spatial_pool_size,
            )
            if config.world_model_stage == "stage2":
                self.register_buffer(
                    "stage2_step",
                    torch.zeros((), dtype=torch.long),
                    persistent=True,
                )

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            if config.world_model_stage == "stage2":
                # Stage 2 does not call the generic ``forward`` method. Compile
                # every fixed-shape neural path that is executed during
                # training. Compile bound ``forward`` methods rather than whole
                # nn.Module objects so checkpoint state-dict keys remain
                # unchanged (i.e. no ``_orig_mod.`` prefix).
                self.encode_wam_prefix_for_training = torch.compile(
                    self.encode_wam_prefix_for_training,
                    mode=config.compile_mode,
                )
                self.forward_expert_with_cached_context = torch.compile(
                    self.forward_expert_with_cached_context,
                    mode=config.compile_mode,
                )
                self.world_action_encoder.forward = torch.compile(
                    self.world_action_encoder.forward,
                    mode=config.compile_mode,
                )
                self.world_state_encoder.forward = torch.compile(
                    self.world_state_encoder.forward,
                    mode=config.compile_mode,
                )
                self.world_model.forward = torch.compile(
                    self.world_model.forward,
                    mode=config.compile_mode,
                )
                self.world_vision_backbone.forward = torch.compile(
                    self.world_vision_backbone.forward,
                    mode=config.compile_mode,
                )
                self.wam_future_condition_resampler.forward = torch.compile(
                    self.wam_future_condition_resampler.forward,
                    mode=config.compile_mode,
                )
                logging.info(
                    "Enabled torch.compile for all WAVE-0 Stage-2 neural paths: "
                    "VLM prefill, cached-context Expert, target vision, GT-action "
                    "teacher, state encoder, latent World Model, and future resampler"
                )
            else:
                self.forward = torch.compile(self.forward, mode=config.compile_mode)

        msg = """An incorrect transformer version is used, please create an issue on https://github.com/huggingface/lerobot/issues"""

        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05Pytorch model")

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def _select_world_model_images(self, images: list[Tensor]) -> list[Tensor]:
        """Select physical cameras from inputs ordered like config.image_features.

        Direct model callers supply the full VLM camera layout, including empty
        slots. The policy entry point instead passes explicitly named world images.
        """
        image_keys = list(self.config.image_features)
        if len(images) != len(image_keys):
            raise ValueError(
                f"Preprocessed image count {len(images)} != image_features {len(image_keys)}."
            )
        world_keys = list(self.config.world_model_image_features)
        if world_keys == image_keys:
            return images
        images_by_key = dict(zip(image_keys, images, strict=True))
        return [images_by_key[key] for key in world_keys]

    @torch.no_grad()
    def encode_world_images(self, images: list[Tensor]) -> Tensor:
        """Extract frozen, spatially pooled PaliGemma image tokens for each camera."""
        target_backbone = getattr(self, "world_vision_backbone", None)
        if target_backbone is not None:
            target_backbone.eval()
        else:
            self.paligemma_with_expert.paligemma.vision_tower.eval()
            self.paligemma_with_expert.paligemma.multi_modal_projector.eval()
        pooled_latents = []
        for image in images:
            image_tokens = (
                target_backbone(image)
                if target_backbone is not None
                else self.paligemma_with_expert.embed_image(image)
            )
            pooled = spatial_pool_image_tokens(
                image_tokens,
                output_size=self.config.world_model_spatial_pool_size,
            )
            pooled_latents.append(pooled.to(dtype=torch.float32))
        return torch.stack(pooled_latents, dim=1).detach()

    @torch.no_grad()
    def encode_world_image_pairs(
        self, current_images: list[Tensor], future_images: list[Tensor]
    ) -> tuple[Tensor, Tensor]:
        """Encode current/future frames together to improve frozen vision throughput."""
        if len(current_images) != len(future_images):
            raise ValueError("Current and future image lists must contain the same cameras.")
        target_backbone = getattr(self, "world_vision_backbone", None)
        if target_backbone is not None:
            target_backbone.eval()
        else:
            self.paligemma_with_expert.paligemma.vision_tower.eval()
            self.paligemma_with_expert.paligemma.multi_modal_projector.eval()
        current_latents = []
        future_latents = []
        for current_image, future_image in zip(current_images, future_images, strict=True):
            batch_size = current_image.shape[0]
            image_batch = torch.cat([current_image, future_image], dim=0)
            image_tokens = (
                target_backbone(image_batch)
                if target_backbone is not None
                else self.paligemma_with_expert.embed_image(image_batch)
            )
            pooled = spatial_pool_image_tokens(
                image_tokens,
                output_size=self.config.world_model_spatial_pool_size,
            ).to(dtype=torch.float32)
            current_latents.append(pooled[:batch_size])
            future_latents.append(pooled[batch_size:])
        return (
            torch.stack(current_latents, dim=1).detach(),
            torch.stack(future_latents, dim=1).detach(),
        )

    def embed_prefix(
        self, images, img_masks, tokens, masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed the original PI0.5 image/language context.

        This method deliberately contains no WAM latent-action queries. The
        Action Expert continues to receive the original PI0.5 context KV; the
        queries are internal VLM readout tokens created by ``embed_wam_prefix``.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_wam_prefix(self, images, img_masks, tokens, masks) -> tuple[Tensor, Tensor, Tensor]:
        """Append an internal latent-action query block to PI0.5 context.

        The attention-block markers are ``0`` for image/language context and
        ``[1, 0, ..., 0]`` for the query block. Context tokens therefore cannot
        read the learned queries, while every query can read the full context
        and all other queries. Query KV is removed before conditioning the
        Action Expert; only the predicted future-state condition is added.
        """
        context_embs, context_pad_masks, context_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks
        )
        bsize = context_embs.shape[0]
        latent_action_queries = self.wam_latent_action_queries.expand(bsize, -1, -1)
        latent_action_queries = latent_action_queries.to(
            device=context_embs.device,
            dtype=context_embs.dtype,
        )
        query_pad_masks = torch.ones(
            bsize,
            self.config.wam_latent_action_num_queries,
            dtype=torch.bool,
            device=context_pad_masks.device,
        )
        query_att_masks = torch.zeros_like(query_pad_masks)
        query_att_masks[:, 0] = True

        return (
            torch.cat([context_embs, latent_action_queries], dim=1),
            torch.cat([context_pad_masks, query_pad_masks], dim=1),
            torch.cat([context_att_masks, query_att_masks], dim=1),
        )

    def encode_wam_prefix(
        self,
        images,
        img_masks,
        tokens,
        masks,
        *,
        use_cache: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Any]:
        """Run the VLM prefix and return its explicit latent-action tokens.

        Returns:
            ``prefix_output``:
                Final VLM hidden states for context and latent-action queries.
            ``latent_action_tokens``:
                Projected final query states with shape
                ``[B, wam_latent_action_num_queries, world_model_width]``.
            ``prefix_pad_masks`` / ``prefix_att_masks``:
                The 1-D masks needed by the following Action Expert pass.
            ``past_key_values``:
                Optional per-layer VLM KV cache used by inference.
        """
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_wam_prefix(
            images, img_masks, tokens, masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (  # noqa: SLF001
            "eager"
        )
        (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=use_cache,
        )
        if prefix_output is None:
            raise RuntimeError("WAVE-0 VLM prefix forward returned no hidden states.")

        query_count = self.config.wam_latent_action_num_queries
        query_output = prefix_output[:, -query_count:].to(dtype=torch.float32)
        latent_action_tokens = self.wam_latent_action_projection(query_output)
        expected_shape = (
            prefix_output.shape[0],
            query_count,
            self.config.world_model_width,
        )
        if latent_action_tokens.shape != expected_shape:
            raise RuntimeError(
                "WAVE-0 latent-action projection returned an invalid shape: "
                f"expected {expected_shape}, got {tuple(latent_action_tokens.shape)}."
            )

        return (
            prefix_output,
            latent_action_tokens,
            prefix_pad_masks,
            prefix_att_masks,
            past_key_values,
        )

    def encode_wam_prefix_for_training(
        self,
        images,
        img_masks,
        tokens,
        masks,
    ) -> tuple[Tensor, Tensor, list[tuple[Tensor, Tensor]]]:
        """Run the Stage-2 VLM exactly once and expose differentiable context KV.

        The returned KV contains only the original PI0.5 image/language context.
        Latent-action query KV is intentionally excluded so the Action Expert can
        consume query information only through the predicted-future bottleneck.
        """
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_wam_prefix(
            images,
            img_masks,
            tokens,
            masks,
        )
        query_count = self.config.wam_latent_action_num_queries
        context_length = prefix_embs.shape[1] - query_count
        if context_length <= 0:
            raise ValueError(
                "WAM training prefix must contain original PI0.5 context before "
                f"the {query_count} latent-action queries."
            )

        prefix_attention = self._prepare_attention_masks_4d(
            make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        )
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        vlm_dtype = self.paligemma_with_expert.paligemma.language_model.layers[
            0
        ].self_attn.q_proj.weight.dtype
        prefix_output, context_key_values = self.paligemma_with_expert.forward_vlm_with_context_kv(
            prefix_embeds=prefix_embs.to(dtype=vlm_dtype),
            attention_mask=prefix_attention,
            position_ids=prefix_position_ids,
            context_length=context_length,
        )

        query_output = prefix_output[:, -query_count:].to(dtype=torch.float32)
        latent_action_tokens = self.wam_latent_action_projection(query_output)
        expected_shape = (
            prefix_output.shape[0],
            query_count,
            self.config.world_model_width,
        )
        if latent_action_tokens.shape != expected_shape:
            raise RuntimeError(
                "WAVE-0 training latent-action projection returned an invalid shape: "
                f"expected {expected_shape}, got {tuple(latent_action_tokens.shape)}."
            )
        return (
            latent_action_tokens,
            prefix_pad_masks[:, :context_length],
            context_key_values,
        )

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self,
        images,
        img_masks,
        tokens,
        masks,
        actions,
        noise=None,
        time=None,
        return_intermediates: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Do a full training forward pass and compute the loss."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        losses = F.mse_loss(u_t, v_t, reduction="none")
        if return_intermediates:
            # x_t = t * noise + (1 - t) * action and v = dx/dt, hence
            # the one-step clean-action estimate at t=0 is x_t - t * v.
            predicted_clean_actions = recover_clean_actions(x_t, v_t, time)
            return losses, {
                "x_t": x_t,
                "time": time,
                "predicted_velocity": v_t,
                "predicted_clean_actions": predicted_clean_actions,
            }
        return losses

    def forward_expert_with_cached_context(
        self,
        context_pad_masks: Tensor,
        context_key_values: list[tuple[Tensor, Tensor]],
        actions: Tensor,
        future_condition: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        """Compute flow loss without recomputing the VLM image/language prefix."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        target_velocity = noise - actions
        action_embeds, action_pad_masks, action_att_masks, action_adarms_cond = self.embed_suffix(x_t, time)

        batch_size = actions.shape[0]
        expected_future_tokens = (
            len(self.config.world_model_image_features)
            * self.config.wam_future_condition_spatial_pool_size**2
        )
        expected_future_shape = (
            batch_size,
            expected_future_tokens,
            self.action_in_proj.out_features,
        )
        if future_condition.shape != expected_future_shape:
            raise ValueError(
                "Future condition has an invalid shape; "
                f"expected {expected_future_shape}, got {tuple(future_condition.shape)}."
            )
        if context_pad_masks.ndim != 2 or context_pad_masks.shape[0] != batch_size:
            raise ValueError(
                "Context pad mask must have shape [B,C]; "
                f"got {tuple(context_pad_masks.shape)} for batch size {batch_size}."
            )

        future_pad_masks = torch.ones(
            batch_size,
            expected_future_tokens,
            dtype=torch.bool,
            device=actions.device,
        )
        future_att_masks = torch.zeros_like(future_pad_masks)
        future_att_masks[:, 0] = True
        future_adarms_cond = torch.zeros_like(action_adarms_cond)

        expert_dtype = self.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype
        future_condition = future_condition.to(dtype=expert_dtype)
        action_embeds = action_embeds.to(dtype=expert_dtype)

        # Construct the same blockwise visibility as the original three-stream
        # forward, then retain only the future/action query rows. Context KV is
        # supplied explicitly and therefore has no query rows in this pass.
        context_att_masks = torch.zeros_like(context_pad_masks)
        pad_masks = torch.cat(
            [context_pad_masks, future_pad_masks, action_pad_masks],
            dim=1,
        )
        att_masks = torch.cat(
            [
                context_att_masks,
                future_att_masks,
                action_att_masks.to(dtype=torch.bool),
            ],
            dim=1,
        )
        full_attention = make_att_2d_masks(pad_masks, att_masks)
        context_length = context_pad_masks.shape[1]
        expert_attention = self._prepare_attention_masks_4d(full_attention[:, context_length:, :])
        full_position_ids = torch.cumsum(pad_masks, dim=1) - 1
        expert_position_ids = full_position_ids[:, context_length:]

        _, action_output = self.paligemma_with_expert.forward_expert_with_context_kv(
            context_key_values=context_key_values,
            future_embeds=future_condition,
            action_embeds=action_embeds,
            attention_mask=expert_attention,
            position_ids=expert_position_ids,
            future_adarms_cond=future_adarms_cond,
            action_adarms_cond=action_adarms_cond,
        )
        action_output = action_output[:, -self.config.chunk_size :].float()
        predicted_velocity = self.action_out_proj(action_output)
        return F.mse_loss(target_velocity, predicted_velocity, reduction="none")

    @staticmethod
    def _crop_query_tokens_from_cache(past_key_values: Any, query_count: int) -> Any:
        """Remove latent-action query KV while preserving the original PI0.5 KV."""
        if past_key_values is None:
            raise RuntimeError("VLM did not return a KV cache for WAM inference.")
        if hasattr(past_key_values, "crop"):
            sequence_length = past_key_values.get_seq_length()
            if sequence_length < query_count:
                raise RuntimeError(
                    f"Cannot crop {query_count} query tokens from cache length {sequence_length}."
                )
            past_key_values.crop(-query_count)
            return past_key_values

        cropped_layers = []
        for layer in past_key_values:
            if len(layer) < 2:
                raise RuntimeError("Unsupported legacy KV-cache layer format.")
            key, value, *rest = layer
            cropped_layers.append(
                (
                    key[..., :-query_count, :],
                    value[..., :-query_count, :],
                    *rest,
                )
            )
        return tuple(cropped_layers)

    def _prefill_future_condition_cache(
        self,
        *,
        prefix_pad_masks: Tensor,
        past_key_values: Any,
        future_condition: Tensor,
    ) -> tuple[Tensor, Any]:
        """Append fixed predicted-future Expert KV once before denoising."""
        batch_size, future_length, _ = future_condition.shape
        expert_dtype = self.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype
        future_pad_masks = torch.ones(
            batch_size,
            future_length,
            dtype=torch.bool,
            device=future_condition.device,
        )
        future_att_masks = torch.zeros_like(future_pad_masks)
        future_att_masks[:, 0] = True

        prefix_attention = prefix_pad_masks[:, None, :].expand(
            batch_size,
            future_length,
            prefix_pad_masks.shape[1],
        )
        future_attention = make_att_2d_masks(
            future_pad_masks,
            future_att_masks,
        )
        attention_mask = self._prepare_attention_masks_4d(
            torch.cat([prefix_attention, future_attention], dim=2)
        ).to(dtype=expert_dtype)
        prefix_offsets = prefix_pad_masks.sum(dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(future_pad_masks, dim=1) - 1
        future_adarms_cond = torch.zeros(
            batch_size,
            self.action_in_proj.out_features,
            dtype=future_condition.dtype,
            device=future_condition.device,
        )
        _, augmented_cache = self.paligemma_with_expert.forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, future_condition.to(dtype=expert_dtype)],
            use_cache=True,
            adarms_cond=[None, future_adarms_cond],
        )
        if augmented_cache is None:
            raise RuntimeError("Action Expert did not return future-condition KV cache.")
        return (
            torch.cat([prefix_pad_masks, future_pad_masks], dim=1),
            augmented_cache,
        )

    @torch.no_grad()
    def _prepare_wam_inference_context(
        self,
        images,
        img_masks,
        tokens,
        masks,
        state,
        world_images=None,
    ) -> tuple[Tensor, Any]:
        """Compute latent action, predicted future state, and reusable Expert KV."""
        (
            _,
            latent_action_tokens,
            wam_prefix_pad_masks,
            _,
            past_key_values,
        ) = self.encode_wam_prefix(
            images,
            img_masks,
            tokens,
            masks,
            use_cache=True,
        )
        query_count = self.config.wam_latent_action_num_queries
        past_key_values = self._crop_query_tokens_from_cache(
            past_key_values,
            query_count,
        )
        prefix_pad_masks = wam_prefix_pad_masks[:, :-query_count]

        # VLM inputs can include masked camera slots absent from world-model training.
        if world_images is None:
            world_images = self._select_world_model_images(images)
        current_latent = self.encode_world_images(world_images)
        state = pad_vector(state, self.config.max_state_dim).to(
            device=current_latent.device,
            dtype=torch.float32,
        )
        state_token = self.world_state_encoder(state)
        predicted_delta = self.world_model(
            current_latent,
            state_token,
            latent_action_tokens.to(dtype=torch.float32),
        )
        predicted_future = current_latent + predicted_delta
        future_condition = self.wam_future_condition_resampler(predicted_future)
        return self._prefill_future_condition_cache(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            future_condition=future_condition,
        )

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)

        if self.config.world_model_stage in {"stage2", "inference"}:
            state = kwargs.pop("state", None)
            if state is None:
                raise ValueError("WAVE-0 inference requires observation.state.")
            prefix_pad_masks, past_key_values = self._prepare_wam_inference_context(
                images,
                img_masks,
                tokens,
                masks,
                state,
                world_images=kwargs.pop("world_images", None),
            )
        else:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images,
                img_masks,
                tokens,
                masks,
            )
            prefix_att_2d_masks = make_att_2d_masks(
                prefix_pad_masks,
                prefix_att_masks,
            )
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
            self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (  # noqa: SLF001
                "eager"
            )
            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)
        expert_dtype = self.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.q_proj.weight.dtype
        suffix_embs = suffix_embs.to(dtype=expert_dtype)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks).to(dtype=expert_dtype)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class WAVE0Policy(PreTrainedPolicy):
    """WAVE-0: a PI0.5 policy conditioned on predicted future visual latents."""

    config_class = WAVE0Config
    name = "wave0"

    def __init__(
        self,
        config: WAVE0Config,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize the core PI05 model
        self.init_rtc_processor()
        if config.world_model_stage == "stage1":
            self.model = WAVE0Stage1Model(config)
        else:
            self.model = PI05Pytorch(config, rtc_processor=self.rtc_processor)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing and config.world_model_stage != "stage1":
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)
        self._configure_trainability()

        self.reset()

    def _configure_trainability(self) -> None:
        """Apply the stage-specific freeze boundary before optimizer construction."""
        if self.config.world_model_stage == "stage1":
            self.model.requires_grad_(False)
            self.model.world_action_encoder.requires_grad_(True)
            self.model.world_state_encoder.requires_grad_(True)
            self.model.world_model.requires_grad_(True)
        elif self.config.world_model_stage == "stage2":
            # The World Model is a differentiable supervisor: parameters stay
            # frozen, while gradients with respect to its action input are kept.
            self.model.world_action_encoder.requires_grad_(False)
            self.model.world_state_encoder.requires_grad_(False)
            self.model.world_model.requires_grad_(False)
            self.model.world_vision_backbone.requires_grad_(False)
        elif self.config.world_model_stage == "inference":
            self.model.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.world_model_stage == "stage1":
            # Frozen PI0.5 is a deterministic target encoder during Stage 1.
            self.model.vision_backbone.eval()
            self.model.world_action_encoder.train(mode)
            self.model.world_state_encoder.train(mode)
            self.model.world_model.train(mode)
        elif self.config.world_model_stage == "stage2":
            self.model.world_action_encoder.eval()
            self.model.world_state_encoder.eval()
            self.model.world_model.eval()
            self.model.world_vision_backbone.eval()
        elif self.config.world_model_stage == "inference":
            self.model.world_state_encoder.eval()
            self.model.world_model.eval()
            self.model.world_vision_backbone.eval()
        return self

    @staticmethod
    def _resolve_safetensors_file(
        pretrained_name_or_path: str | Path,
        *,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> str:
        model_path = Path(pretrained_name_or_path)
        if model_path.is_file():
            return str(model_path)
        if model_path.is_dir():
            model_file = model_path / "model.safetensors"
            if not model_file.is_file():
                raise FileNotFoundError(f"model.safetensors not found in {model_path}.")
            return str(model_file)

        from transformers.utils import cached_file

        resolved = cached_file(
            str(pretrained_name_or_path),
            "model.safetensors",
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            revision=revision,
            local_files_only=local_files_only,
        )
        if resolved is None:
            raise FileNotFoundError(f"Could not resolve model.safetensors from {pretrained_name_or_path}.")
        return resolved

    @staticmethod
    def _checkpoint_contains_world_modules(model_file: str) -> bool:
        from safetensors import safe_open

        with safe_open(model_file, framework="pt", device="cpu") as checkpoint:
            return any(
                key.startswith(("model.world_action_encoder.", "world_action_encoder."))
                for key in checkpoint.keys()
            )

    @staticmethod
    def _checkpoint_contains_vision_backbone(model_file: str) -> bool:
        from safetensors import safe_open

        vision_markers = (
            "paligemma_with_expert.paligemma.model.vision_tower.",
            "model.paligemma_with_expert.paligemma.model.vision_tower.",
        )
        with safe_open(model_file, framework="pt", device="cpu") as checkpoint:
            return any(key.startswith(vision_markers) for key in checkpoint.keys())

    @staticmethod
    def _load_frozen_vision_backbone(backbone: nn.Module, model_file: str) -> None:
        """Selectively read vision tower/projector tensors into a frozen target encoder."""
        from safetensors import safe_open

        source_prefixes = (
            "paligemma_with_expert.paligemma.model.",
            "model.paligemma_with_expert.paligemma.model.",
        )
        expected_keys = set(backbone.state_dict())
        vision_state = {}
        with safe_open(model_file, framework="pt", device="cpu") as checkpoint:
            for source_key in checkpoint.keys():
                target_key = None
                for prefix in source_prefixes:
                    if source_key.startswith(prefix):
                        candidate = source_key.removeprefix(prefix)
                        if candidate.startswith(("vision_tower.", "multi_modal_projector.")):
                            target_key = candidate
                        break
                if target_key in expected_keys:
                    vision_state[target_key] = checkpoint.get_tensor(source_key)

        missing = sorted(expected_keys - set(vision_state))
        unexpected = sorted(set(vision_state) - expected_keys)
        if missing or unexpected:
            raise RuntimeError(
                "Selective PI0.5 vision loading failed. "
                f"Missing keys: {missing[:10]}; unexpected keys: {unexpected[:10]}."
            )
        backbone.load_state_dict(vision_state, strict=True)
        backbone.requires_grad_(False)
        backbone.eval()
        logging.info("Loaded %d frozen PI0.5 vision tensors from %s", len(vision_state), model_file)

    @classmethod
    def _load_stage1_vision_backbone(cls, model: "WAVE0Policy", model_file: str) -> None:
        """Load the minimal Stage-1 model's frozen target encoder."""
        cls._load_frozen_vision_backbone(model.model.vision_backbone, model_file)

    @classmethod
    def _load_stage2_vision_backbone(cls, model: "WAVE0Policy", model_file: str) -> None:
        """Load Stage 2's vision target encoder, separate from trainable policy vision."""
        cls._load_frozen_vision_backbone(model.model.world_vision_backbone, model_file)

    @staticmethod
    def _load_stage1_world_modules(model: "WAVE0Policy", model_file: str) -> None:
        """Load the three trainable Stage-1 modules from a world-only checkpoint."""
        from safetensors import safe_open

        module_names = ("world_action_encoder", "world_state_encoder", "world_model")
        states: dict[str, dict[str, Tensor]] = {name: {} for name in module_names}
        with safe_open(model_file, framework="pt", device="cpu") as checkpoint:
            for source_key in checkpoint.keys():
                normalized_key = source_key.removeprefix("model.")
                for module_name in module_names:
                    prefix = f"{module_name}."
                    if normalized_key.startswith(prefix):
                        states[module_name][normalized_key.removeprefix(prefix)] = checkpoint.get_tensor(
                            source_key
                        )
                        break

        for module_name in module_names:
            module = getattr(model.model, module_name)
            if not states[module_name]:
                raise RuntimeError(f"Stage-1 checkpoint has no tensors for {module_name}.")
            module.load_state_dict(states[module_name], strict=True)
        logging.info("Loaded Stage-1 world modules from %s", model_file)

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save Stage 1 as a compact world-only checkpoint; never duplicate frozen vision."""
        if self.config.world_model_stage != "stage1":
            super()._save_pretrained(save_directory)
            if self.config.wam_share_frozen_vision_encoder:
                from safetensors import safe_open

                model_file = save_directory / "model.safetensors"
                with safe_open(model_file, framework="pt", device="cpu") as checkpoint:
                    duplicated = [key for key in checkpoint.keys() if "world_vision_backbone" in key]
                if duplicated:
                    raise RuntimeError(
                        "Shared vision checkpoint unexpectedly contains duplicate World Model "
                        f"vision tensors: {duplicated[:5]}"
                    )
                logging.info(
                    "Saved shared-vision WAVE-0 checkpoint with one policy-owned vision copy to %s",
                    save_directory,
                )
            return
        if not self.config.world_model_base_model_path:
            raise RuntimeError(
                "world_model_base_model_path is missing; cannot save a restorable Stage-1 checkpoint."
            )

        from safetensors.torch import save_file

        self.config._save_pretrained(save_directory)
        world_state: dict[str, Tensor] = {}
        for module_name in ("world_action_encoder", "world_state_encoder", "world_model"):
            module = getattr(self.model, module_name)
            for key, value in module.state_dict().items():
                world_state[f"model.{module_name}.{key}"] = value.detach().cpu().contiguous()
        save_file(
            world_state,
            str(save_directory / "model.safetensors"),
            metadata={
                "format": "pt",
                "wave0_checkpoint_type": "stage1_world_only",
                "base_model_path": str(self.config.world_model_base_model_path),
            },
        )
        logging.info(
            "Saved %d Stage-1 world tensors without the frozen vision backbone to %s",
            len(world_state),
            save_directory,
        )

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping and display important disclaimer."""
        print(
            "WAVE-0 extends the LeRobot PI0.5 / OpenPI implementation. \n"
            "The PI0.5 backbone retains the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Calling WAVE0Policy directly on an official PI0.5 folder yields a
        # PI05Config. Convert the shared fields so Stage 1 can still be created.
        if not isinstance(config, WAVE0Config):
            # v0.4.4 PreTrainedConfig has no to_dict(). Keep nested PolicyFeature
            # objects intact instead of flattening them with dataclasses.asdict.
            source_config = {
                field.name: getattr(config, field.name) for field in dataclasses.fields(config) if field.init
            }
            init_fields = {field.name for field in dataclasses.fields(WAVE0Config) if field.init}
            config = WAVE0Config(**{key: value for key, value in source_config.items() if key in init_fields})

        if config.world_model_stage == "stage1":
            checkpoint_file = cls._resolve_safetensors_file(
                pretrained_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                revision=revision,
                local_files_only=local_files_only,
            )
            contains_world = cls._checkpoint_contains_world_modules(checkpoint_file)
            contains_vision = cls._checkpoint_contains_vision_backbone(checkpoint_file)

            if contains_vision:
                vision_checkpoint_file = checkpoint_file
                if not config.world_model_base_model_path:
                    config.world_model_base_model_path = str(pretrained_name_or_path)
            else:
                if not config.world_model_base_model_path:
                    raise RuntimeError(
                        "This is a world-only Stage-1 checkpoint, but its config does not contain "
                        "world_model_base_model_path."
                    )
                vision_checkpoint_file = cls._resolve_safetensors_file(
                    config.world_model_base_model_path,
                    cache_dir=cache_dir,
                    force_download=False,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    revision=revision,
                    local_files_only=local_files_only,
                )

            model = cls(config, **kwargs)
            cls._load_stage1_vision_backbone(model, vision_checkpoint_file)
            if contains_world:
                cls._load_stage1_world_modules(model, checkpoint_file)
            model.eval()
            return model

        if config.world_model_stage == "stage2" and not config.world_model_base_model_path:
            stage1_config = PreTrainedConfig.from_pretrained(
                config.world_model_stage1_checkpoint_path,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            config.world_model_base_model_path = getattr(stage1_config, "world_model_base_model_path", None)
            if not config.world_model_base_model_path:
                raise RuntimeError(
                    "Could not recover world_model_base_model_path from the Stage-1 checkpoint. "
                    "Set it explicitly to the PI0.5 checkpoint used by Stage 1."
                )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Now manually load and remap the state dict
        try:
            # Try to load the pytorch_model.bin or model.safetensors file
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                # Try safetensors first
                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    revision=revision,
                    local_files_only=local_files_only,
                )
                if config.world_model_stage in {"disabled", "inference"}:
                    # Avoid allocating or even reading training-only World Model
                    # tensors during policy deployment. WAM inference still
                    # keeps the World Model, state/current-image encoders, and
                    # future-condition modules, but skips the GT-action teacher.
                    from safetensors import safe_open

                    with safe_open(resolved_file, framework="pt", device="cpu") as checkpoint:
                        all_keys = list(checkpoint.keys())
                        if config.world_model_stage == "disabled":
                            policy_keys = [key for key in all_keys if not _is_world_model_state_key(key)]
                        else:
                            policy_keys = [key for key in all_keys if not _is_wam_teacher_state_key(key)]
                        original_state_dict = {key: checkpoint.get_tensor(key) for key in policy_keys}
                    print(
                        "✓ Selectively loaded state dict from model.safetensors "
                        f"(skipped {len(all_keys) - len(policy_keys)} deployment-unused tensors)"
                    )
                else:
                    from safetensors.torch import load_file

                    original_state_dict = load_file(resolved_file)
                    print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                raise RuntimeError(
                    f"Could not resolve model.safetensors from {pretrained_name_or_path}: {e}"
                ) from e

            # First, fix any key differences # see openpi `model.py, _fix_pytorch_state_dict_keys`
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                    if remap_count <= 10:  # Only print first 10 to avoid spam
                        print(f"Remapped: {key} -> {new_key}")
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            if config.world_model_stage == "disabled":
                # Defensive filtering for any key spelling introduced by the
                # compatibility remapper.
                remapped_state_dict = {
                    key: value
                    for key, value in remapped_state_dict.items()
                    if not _is_world_model_state_key(key)
                }
            elif config.world_model_stage == "inference":
                remapped_state_dict = {
                    key: value
                    for key, value in remapped_state_dict.items()
                    if not _is_wam_teacher_state_key(key)
                }
            if config.wam_share_frozen_vision_encoder:
                # Older checkpoints may contain a separate frozen target copy.
                # The shared layout always restores vision from the policy-owned
                # keys, so discard the redundant legacy keys before loading.
                remapped_state_dict = {
                    key: value
                    for key, value in remapped_state_dict.items()
                    if not key.startswith("model.world_vision_backbone.")
                }

            # Load with ``strict=False`` first because an official PI0.5 checkpoint
            # intentionally has no Stage-1 world-model parameters. Strictness is
            # applied below after whitelisting only these new module prefixes.
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=False)

            allowed_missing_prefixes = (
                "model.world_action_encoder.",
                "model.world_state_encoder.",
                "model.world_model.",
                "model.world_vision_backbone.",
                "model.stage2_step",
                "model.wam_latent_action_queries",
                "model.wam_latent_action_projection.",
                "model.wam_future_condition_resampler.",
            )
            invalid_missing_keys = [
                key for key in missing_keys if not key.startswith(allowed_missing_prefixes)
            ]
            if strict and (invalid_missing_keys or unexpected_keys):
                raise RuntimeError(
                    "WAVE-0 checkpoint loading found incompatible keys. "
                    f"Missing non-world keys: {invalid_missing_keys[:10]}; "
                    f"unexpected keys: {unexpected_keys[:10]}."
                )

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            raise RuntimeError(f"Could not load WAVE-0 pretrained weights: {e}") from e

        if config.world_model_stage == "stage2":
            stage1_model_file = cls._resolve_safetensors_file(
                config.world_model_stage1_checkpoint_path,
                cache_dir=cache_dir,
                force_download=False,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                revision=revision,
                local_files_only=local_files_only,
            )
            cls._load_stage1_world_modules(model, stage1_model_file)
            if config.wam_share_frozen_vision_encoder:
                logging.info(
                    "World Model target vision shares the frozen policy vision tower/projector; "
                    "no duplicate target vision weights were loaded."
                )
            else:
                target_vision_file = cls._resolve_safetensors_file(
                    config.world_model_base_model_path,
                    cache_dir=cache_dir,
                    force_download=False,
                    resume_download=resume_download,
                    proxies=proxies,
                    token=token,
                    revision=revision,
                    local_files_only=local_files_only,
                )
                cls._load_stage2_vision_backbone(model, target_vision_file)
            model._configure_trainability()

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi05
            # pi05 model expects time_mlp_*, but checkpoint might have action_time_mlp_*
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            # Also handle state_proj which shouldn't exist in pi05
            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
                continue

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            # Safetensors stores tied PaliGemma lm_head/embed_tokens weights only once.
            # Restore the alias expected by strict state-dict loading.
            if key in {
                "model.paligemma_with_expert.paligemma.lm_head.weight",
                "paligemma_with_expert.paligemma.lm_head.weight",
            }:
                fixed_state_dict[
                    "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
                ] = value.clone()

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("WAVE-0 has no trainable parameters for the selected stage.")
        return parameters

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _preprocess_image_tensor(self, image: Tensor) -> Tensor:
        """Apply the exact PI0.5 image resize and SigLIP normalization to one camera."""
        if image.ndim != 4:
            raise ValueError(f"Expected one batched image tensor [B,C,H,W] or [B,H,W,C], got {image.shape}.")

        device = next(self.parameters()).device
        image = image.to(device=device, dtype=torch.float32)
        is_channels_first = image.shape[1] == 3
        if is_channels_first:
            image = image.permute(0, 2, 3, 1)
        elif image.shape[-1] != 3:
            raise ValueError(f"Could not identify the RGB channel in image shape {tuple(image.shape)}.")

        if image.shape[1:3] != self.config.image_resolution:
            image = resize_with_pad_torch(image, *self.config.image_resolution)
        image = image * 2.0 - 1.0
        if is_channels_first:
            image = image.permute(0, 3, 1, 2)
        return image

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = self._preprocess_image_tensor(batch[key])

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def _prepare_world_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor], Tensor]:
        """Prepare current/future camera pairs and their per-camera validity mask."""
        horizon = self.config.world_model_future_horizons[0]
        current_images: list[Tensor] = []
        future_images: list[Tensor] = []
        future_valid_masks: list[Tensor] = []

        for image_key in self.config.world_model_image_features:
            if image_key not in batch:
                raise ValueError(f"Stage-1 batch is missing current camera {image_key!r}.")
            future_key = world_model_future_image_key(image_key, horizon)
            if future_key not in batch:
                raise ValueError(
                    f"Stage-1 batch is missing {future_key!r}. Ensure WAVE-0 per-camera "
                    "delta timestamps and the future-image processor are enabled."
                )

            current_images.append(self._preprocess_image_tensor(batch[image_key]))
            future_images.append(self._preprocess_image_tensor(batch[future_key]))

            pad_key = f"{image_key}_is_pad"
            if pad_key in batch:
                pad_mask = batch[pad_key].to(device=current_images[-1].device, dtype=torch.bool)
                if pad_mask.ndim != 2 or pad_mask.shape[1] != 2:
                    raise ValueError(f"Expected {pad_key!r} with shape [B,2], got {tuple(pad_mask.shape)}.")
                future_valid_masks.append(~pad_mask[:, 1])
            else:
                future_valid_masks.append(
                    torch.ones(
                        current_images[-1].shape[0],
                        dtype=torch.bool,
                        device=current_images[-1].device,
                    )
                )

        return current_images, future_images, torch.stack(future_valid_masks, dim=1)

    @staticmethod
    def _same_task_action_permutation(tasks, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Create a cyclic non-identity permutation independently within each task."""
        permutation = torch.arange(batch_size, device=device)
        valid = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if tasks is None:
            if batch_size > 1:
                permutation = torch.roll(permutation, shifts=1)
                valid.fill_(True)
            return permutation, valid

        groups: dict[str, list[int]] = {}
        for index, task in enumerate(tasks):
            groups.setdefault(str(task), []).append(index)
        for indices in groups.values():
            if len(indices) < 2:
                continue
            destination = torch.tensor(indices, dtype=torch.long, device=device)
            source = torch.roll(destination, shifts=1)
            permutation[destination] = source
            valid[destination] = True
        return permutation, valid

    @classmethod
    def _make_negative_actions(cls, actions: Tensor, tasks) -> tuple[Tensor, Tensor]:
        """Build task-preserving negatives using peers or a temporal trajectory roll.

        RoboTwin stores many paraphrased instructions, so exact instruction groups can
        be singletons. Those samples use a half-horizon temporal roll of their own GT
        trajectory instead of an unrelated task's action.
        """
        permutation, peer_valid = cls._same_task_action_permutation(tasks, actions.shape[0], actions.device)
        negative_actions = actions[permutation].clone()
        temporal_roll = torch.roll(actions, shifts=max(1, actions.shape[1] // 2), dims=1)
        negative_actions[~peer_valid] = temporal_roll[~peer_valid]
        changed = (negative_actions - actions).abs().flatten(1).amax(dim=1) > 1e-6
        return negative_actions, changed

    @staticmethod
    def _masked_per_sample_mean(values: Tensor, camera_valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Reduce [B,C,P] values while excluding padded future-camera targets."""
        mask = camera_valid_mask[:, :, None].expand_as(values)
        denominator = mask.sum(dim=(1, 2))
        per_sample = (values * mask).sum(dim=(1, 2)) / denominator.clamp_min(1)
        return per_sample, denominator > 0

    @staticmethod
    def _mean_over_valid_samples(values: Tensor, valid: Tensor) -> Tensor:
        if valid.any():
            return values[valid].mean()
        return values.sum() * 0.0

    def forward_world_model(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
        compute_diagnostics: bool = True,
    ) -> tuple[Tensor, dict]:
        """Stage-1 GT-action latent world-model training forward."""
        current_images, future_images, future_valid_mask = self._prepare_world_images(batch)
        current_latent, future_latent = self.model.encode_world_image_pairs(current_images, future_images)

        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim).to(
            device=current_latent.device, dtype=torch.float32
        )
        action_state = pad_vector(batch[OBS_STATE], self.config.max_action_dim).to(
            device=current_latent.device, dtype=torch.float32
        )
        horizon = self.config.world_model_future_horizons[0]
        actions = self.prepare_action(batch)[:, : horizon + 1].to(
            device=current_latent.device, dtype=torch.float32
        )

        action_tokens = self.model.world_action_encoder(actions, action_state)
        state_token = self.model.world_state_encoder(state)
        pred_delta = self.model.world_model(current_latent, state_token, action_tokens)
        pred_future = current_latent + pred_delta
        target_delta = future_latent - current_latent

        future_distance_tokens = 1.0 - F.cosine_similarity(pred_future.float(), future_latent.float(), dim=-1)
        future_per_sample, future_sample_valid = self._masked_per_sample_mean(
            future_distance_tokens, future_valid_mask
        )
        delta_tokens = F.smooth_l1_loss(pred_delta.float(), target_delta.float(), reduction="none").mean(
            dim=-1
        )
        delta_per_sample, _ = self._masked_per_sample_mean(delta_tokens, future_valid_mask)

        negative_actions, rank_sample_valid = self._make_negative_actions(
            actions, batch.get("world_model_task", batch.get("task"))
        )
        rank_sample_valid = rank_sample_valid & future_sample_valid
        wrong_per_sample = future_per_sample.detach()
        rank_per_sample = torch.zeros_like(future_per_sample)
        if self.config.world_model_rank_loss_weight > 0 and rank_sample_valid.any():
            wrong_action_tokens = self.model.world_action_encoder(negative_actions, action_state)
            wrong_delta = self.model.world_model(current_latent, state_token, wrong_action_tokens)
            wrong_future = current_latent + wrong_delta
            wrong_distance_tokens = 1.0 - F.cosine_similarity(
                wrong_future.float(), future_latent.float(), dim=-1
            )
            wrong_per_sample, _ = self._masked_per_sample_mean(wrong_distance_tokens, future_valid_mask)
            rank_per_sample = F.relu(
                self.config.world_model_rank_margin + future_per_sample - wrong_per_sample
            )
            rank_per_sample = rank_per_sample * rank_sample_valid

        total_per_sample = (
            self.config.world_model_future_loss_weight * future_per_sample
            + self.config.world_model_delta_loss_weight * delta_per_sample
            + self.config.world_model_rank_loss_weight * rank_per_sample
        )
        valid_total = future_sample_valid

        total_loss = self._mean_over_valid_samples(total_per_sample, valid_total)

        loss_dict: dict[str, float] = {}
        if compute_diagnostics:
            future_loss = self._mean_over_valid_samples(future_per_sample, future_sample_valid)
            delta_loss = self._mean_over_valid_samples(delta_per_sample, future_sample_valid)
            rank_loss = self._mean_over_valid_samples(rank_per_sample, rank_sample_valid)
            wrong_distance = self._mean_over_valid_samples(wrong_per_sample, rank_sample_valid)
            correct_rank_distance = self._mean_over_valid_samples(future_per_sample, rank_sample_valid)
            loss_dict = {
                "loss": total_loss.item(),
                "loss/world_total": total_loss.item(),
                "loss/world_future": future_loss.item(),
                "loss/world_delta": delta_loss.item(),
                "loss/world_rank": rank_loss.item(),
                "metric/correct_future_distance": future_loss.item(),
                "metric/shuffled_future_distance": wrong_distance.item(),
                "metric/action_sensitivity_gap": (wrong_distance - correct_rank_distance).item(),
                "metric/future_valid_fraction": (future_sample_valid.float().mean().item()),
                "metric/rank_valid_fraction": rank_sample_valid.float().mean().item(),
            }

        if compute_diagnostics and self.config.world_model_compute_zero_action_metric:
            with torch.no_grad():
                zero_action_tokens = self.model.world_action_encoder(torch.zeros_like(actions), action_state)
                zero_delta = self.model.world_model(current_latent, state_token, zero_action_tokens)
                zero_future = current_latent + zero_delta
                zero_distance_tokens = 1.0 - F.cosine_similarity(
                    zero_future.float(), future_latent.float(), dim=-1
                )
                zero_per_sample, _ = self._masked_per_sample_mean(zero_distance_tokens, future_valid_mask)
                zero_distance = self._mean_over_valid_samples(zero_per_sample, future_sample_valid)
            loss_dict["metric/zero_action_future_distance"] = zero_distance.item()

        if reduction == "none":
            return total_per_sample, loss_dict
        return total_loss, loss_dict

    def _world_score_per_sample(
        self,
        pred_delta: Tensor,
        current_latent: Tensor,
        future_latent: Tensor,
        future_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return the Stage-1 regression score without its saturated ranking term."""
        pred_future = current_latent + pred_delta
        target_delta = future_latent - current_latent
        future_distance_tokens = 1.0 - F.cosine_similarity(pred_future.float(), future_latent.float(), dim=-1)
        future_per_sample, sample_valid = self._masked_per_sample_mean(
            future_distance_tokens, future_valid_mask
        )
        delta_tokens = F.smooth_l1_loss(pred_delta.float(), target_delta.float(), reduction="none").mean(
            dim=-1
        )
        delta_per_sample, _ = self._masked_per_sample_mean(delta_tokens, future_valid_mask)
        score = (
            self.config.world_model_future_loss_weight * future_per_sample
            + self.config.world_model_delta_loss_weight * delta_per_sample
        )
        return score, future_per_sample, delta_per_sample, sample_valid

    def _latent_action_alignment_per_sample(
        self,
        predicted_tokens: Tensor,
        teacher_tokens: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Measure how well VLM latent actions match the frozen Stage-1 GT-action encoder.

        Both inputs must contain one token per action from ``t`` through the
        configured future horizon. The teacher input is detached defensively so
        this helper can never update the Stage-1 action encoder.
        """
        if predicted_tokens.shape != teacher_tokens.shape:
            raise ValueError(
                "VLM and teacher latent-action tokens must have identical shapes; "
                f"got {tuple(predicted_tokens.shape)} and {tuple(teacher_tokens.shape)}."
            )
        if predicted_tokens.ndim != 3:
            raise ValueError(
                f"Latent-action tokens must have shape [B,T,D]; got {tuple(predicted_tokens.shape)}."
            )

        teacher_tokens = teacher_tokens.detach()
        predicted_float = predicted_tokens.float()
        teacher_float = teacher_tokens.float()
        cosine_per_sample = (1.0 - F.cosine_similarity(predicted_float, teacher_float, dim=-1)).mean(dim=1)
        huber_per_sample = F.smooth_l1_loss(
            predicted_float,
            teacher_float,
            reduction="none",
        ).mean(dim=(1, 2))
        alignment_per_sample = (
            self.config.wam_alignment_cosine_weight * cosine_per_sample
            + self.config.wam_alignment_huber_weight * huber_per_sample
        )
        return alignment_per_sample, cosine_per_sample, huber_per_sample

    def forward_stage2(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Train pre-expert PI0.5-WAM with flow matching and frozen WAM supervision.

        The WAM branch is:

        ``VLM latent-action queries -> frozen World Model -> future latent loss``.

        The frozen Stage-1 ``world_action_encoder`` supplies a persistent
        GT-action teacher target for the latent-action queries. Neither WAM
        term traverses the Action Expert.
        """
        current_images, future_images, future_valid_mask = self._prepare_world_images(batch)
        current_latent, future_latent = self.model.encode_world_image_pairs(current_images, future_images)

        images, img_masks = self._preprocess_images(batch)
        tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        gt_actions = self.prepare_action(batch).to(dtype=torch.float32)

        horizon = self.config.world_model_future_horizons[0]
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim).to(
            device=current_latent.device, dtype=torch.float32
        )
        action_state = pad_vector(batch[OBS_STATE], self.config.max_action_dim).to(
            device=current_latent.device, dtype=torch.float32
        )
        teacher_actions = gt_actions[:, : horizon + 1].to(
            device=current_latent.device,
            dtype=torch.float32,
        )

        # One differentiable VLM prefill produces both the explicit latent
        # action and per-layer original-context KV. The Expert reuses this KV;
        # image/language context is not encoded a second time.
        (
            predicted_action_tokens,
            context_pad_masks,
            context_key_values,
        ) = self.model.encode_wam_prefix_for_training(
            images,
            img_masks,
            tokens,
            masks,
        )
        predicted_action_tokens = predicted_action_tokens.to(
            device=current_latent.device,
            dtype=torch.float32,
        )

        # Training-only teacher and state condition are fully frozen. The GT
        # action never enters the World Model branch itself.
        with torch.no_grad():
            teacher_action_tokens = self.model.world_action_encoder(
                teacher_actions,
                action_state,
            ).detach()
            state_token = self.model.world_state_encoder(state)

        alignment_per_sample, alignment_cosine_per_sample, alignment_huber_per_sample = (
            self._latent_action_alignment_per_sample(
                predicted_action_tokens,
                teacher_action_tokens,
            )
        )

        # World Model parameters stay frozen, but this call must remain outside
        # no_grad so future/delta losses backpropagate into z_action_vlm.
        predicted_delta = self.model.world_model(current_latent, state_token, predicted_action_tokens)
        predicted_future = current_latent + predicted_delta
        predicted_score, predicted_future_score, predicted_delta_score, sample_valid = (
            self._world_score_per_sample(
                predicted_delta,
                current_latent,
                future_latent,
                future_valid_mask,
            )
        )

        # Flow supervision sees the predicted future through a dedicated Expert
        # condition block. The default detach keeps Flow gradients out of the
        # frozen WAM/VLM latent-action branch while still training the resampler.
        future_condition_source = (
            predicted_future.detach() if self.config.wam_detach_future_condition else predicted_future
        )
        future_condition = self.model.wam_future_condition_resampler(future_condition_source)
        flow_losses = self.model.forward_expert_with_cached_context(
            context_pad_masks,
            context_key_values,
            gt_actions,
            future_condition,
        )
        original_action_dim = self.config.output_features[ACTION].shape[0]
        flow_losses = flow_losses[:, :, :original_action_dim]
        flow_per_sample = flow_losses.mean(dim=(1, 2))

        current_step = int(self.model.stage2_step.item())
        alignment_weight = self.config.get_wam_alignment_weight(current_step)
        wam_per_sample = (predicted_score + alignment_weight * alignment_per_sample) * sample_valid
        wam_loss = self._mean_over_valid_samples(wam_per_sample, sample_valid)
        valid_count = sample_valid.float().sum().clamp_min(1)
        # Preserve λ_wam regardless of how many samples near an episode boundary
        # have an invalid future target.
        wam_per_sample_for_loss = wam_per_sample * (sample_valid.numel() / valid_count)
        warmup_steps = self.config.wam_loss_warmup_steps
        if warmup_steps > 0:
            wam_weight_scale = min(1.0, (current_step + 1) / warmup_steps)
        else:
            wam_weight_scale = 1.0
        effective_wam_weight = self.config.wam_loss_weight * wam_weight_scale
        if self.training:
            self.model.stage2_step.add_(1)
        total_per_sample = flow_per_sample + effective_wam_weight * wam_per_sample_for_loss

        flow_loss = flow_per_sample.mean()
        total_loss = total_per_sample.mean()

        def valid_mean(values: Tensor) -> Tensor:
            return (values * sample_valid).sum() / valid_count

        predicted_norm = predicted_action_tokens.float().norm(dim=-1).mean(dim=1)
        teacher_norm = teacher_action_tokens.float().norm(dim=-1).mean(dim=1)
        norm_ratio = predicted_norm / teacher_norm.clamp_min(1e-6)
        latent_cosine_similarity = 1.0 - alignment_cosine_per_sample
        world_loss = valid_mean(predicted_score)
        alignment_loss = valid_mean(alignment_per_sample)
        alignment_cosine_loss = valid_mean(alignment_cosine_per_sample)
        alignment_huber_loss = valid_mean(alignment_huber_per_sample)
        loss_dict = {
            "loss": total_loss.item(),
            "loss/action_flow": flow_loss.item(),
            "loss/wam_total": wam_loss.item(),
            "loss/wam_total_weighted": (effective_wam_weight * wam_loss).item(),
            "loss/wam_world": world_loss.item(),
            "loss/wam_future": valid_mean(predicted_future_score).item(),
            "loss/wam_delta": valid_mean(predicted_delta_score).item(),
            "loss/wam_alignment": alignment_loss.item(),
            "loss/wam_alignment_cosine": alignment_cosine_loss.item(),
            "loss/wam_alignment_huber": alignment_huber_loss.item(),
            "loss_per_dim": flow_losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
            "metric/wam_alignment_weight": alignment_weight,
            "metric/wam_latent_action_cosine": valid_mean(latent_cosine_similarity).item(),
            "metric/wam_latent_action_norm_ratio": valid_mean(norm_ratio).item(),
            "metric/wam_future_condition_norm": future_condition.float().norm(dim=-1).mean().item(),
            "metric/correct_future_distance": valid_mean(predicted_future_score).item(),
            "metric/future_valid_fraction": sample_valid.float().mean().item(),
            "metric/wam_weight_scale": wam_weight_scale,
            "metric/stage2_step": int(self.model.stage2_step.item()),
        }

        if self.config.wam_compute_shuffled_action_metric:
            with torch.no_grad():
                shuffled_tokens, shuffled_valid = self._make_negative_actions(
                    predicted_action_tokens.detach(),
                    batch.get("world_model_task", batch.get("task")),
                )
                shuffled_valid = shuffled_valid & sample_valid
                shuffled_delta = self.model.world_model(
                    current_latent,
                    state_token,
                    shuffled_tokens,
                )
                shuffled_score, _, _, _ = self._world_score_per_sample(
                    shuffled_delta,
                    current_latent,
                    future_latent,
                    future_valid_mask,
                )
                shuffled_distance = self._mean_over_valid_samples(
                    shuffled_score,
                    shuffled_valid,
                )
                correct_distance = self._mean_over_valid_samples(
                    predicted_score,
                    shuffled_valid,
                )
            loss_dict["metric/wam_shuffled_future_distance"] = shuffled_distance.item()
            loss_dict["metric/wam_action_sensitivity_gap"] = (shuffled_distance - correct_distance).item()
            loss_dict["metric/wam_shuffled_valid_fraction"] = shuffled_valid.float().mean().item()

        if reduction == "none":
            return total_per_sample, loss_dict
        return total_loss, loss_dict

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        if self.config.world_model_stage == "stage1":
            raise RuntimeError(
                "A compact WAVE-0 Stage-1 checkpoint contains no Action Expert and cannot run policy "
                "inference. Load its world modules into the Stage-2 PI0.5 policy first."
            )
        self.eval()

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        if self.config.world_model_stage in {"stage2", "inference"}:
            kwargs["state"] = batch[OBS_STATE]
            # _preprocess_images emits present cameras first, then padded slots.
            # Select by key, not by mask or position, to preserve world-camera order.
            present_keys = [key for key in self.config.image_features if key in batch]
            images_by_key = dict(zip(present_keys, images, strict=False))
            missing_world_keys = [
                key for key in self.config.world_model_image_features if key not in images_by_key
            ]
            if missing_world_keys:
                raise ValueError(f"WAVE-0 inference is missing world-model cameras: {missing_world_keys}")
            kwargs["world_images"] = [images_by_key[key] for key in self.config.world_model_image_features]

        # Sample actions using the model. WAM inference additionally consumes
        # state to predict its future-latent condition once per action chunk.
        actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions

    def forward(
        self,
        batch: dict[str, Tensor],
        reduction: str = "mean",
        compute_diagnostics: bool = True,
    ) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: Training batch containing observations and actions.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        if self.config.world_model_stage == "stage1":
            return self.forward_world_model(
                batch,
                reduction=reduction,
                compute_diagnostics=compute_diagnostics,
            )
        if self.config.world_model_stage == "stage2":
            return self.forward_stage2(batch, reduction=reduction)
        if self.config.world_model_stage == "inference":
            raise RuntimeError(
                "world_model_stage='inference' is deployment-only and cannot be trained. "
                "Use world_model_stage='stage2' for WAM training."
            )

        # Prepare inputs
        images, img_masks = self._preprocess_images(batch)
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.prepare_action(batch)

        # Compute loss (no separate state needed for PI05)
        losses = self.model.forward(images, img_masks, tokens, masks, actions)

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]

        loss_dict = {
            "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
        }

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for PI0.5 fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({common_projections}))"
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }
