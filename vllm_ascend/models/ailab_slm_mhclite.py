# coding=utf-8
## Adapted from
## - AILabSLM huggingface implementation (ailab_slm_mhclite: lite
##   hyper-connections, permutation-basis residual mixing)
## - vllm_ascend/models/ailab_slm_mhc.py
##
## Differences from ailab_slm_mhc:
## - MHCLite replaces the sinkhorn MHC: RMSNorm keeps a learned gamma over
##   the flattened streams, h_res is one softmax over the e! permutation
##   matrices, and hc_post mixes with the transposed h_res
##   (out[o] = post[o]*x + sum_i h_res[i,o]*residual[i])
## - hc_pre runs on the fused LitePreHeads chain op (one extension call);
##   hc_post reuses npu_hc_post (same [in, out] comb convention)
## - params match the HF ailab_slm_mhclite checkpoint (hc_fn.weight /
##   hc_gamma / hc_scale / hc_base, kept fp32)
## - set AILAB_SLM_MHCLITE_NPU_OPS=0 to force the pure-torch path

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
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
import os
import sys
## ailab_slm patch
ailab_slm_local_path = os.getenv("AILAB_SLM_LOCAL_PATH")
if ailab_slm_local_path is not None:
    sys.path.append(ailab_slm_local_path)
else:
    raise RuntimeError("Environment variable `AILAB_SLM_LOCAL_PATH` should be set before running AILabSLM")

from collections.abc import Iterable
from itertools import islice, permutations

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from .ailab_slm import AILabSLMAttention, AILabSLMMLP
from ..ops import mhclite as mhclite_ops


def _npu_mhclite_ops_available() -> bool:
    if os.getenv("AILAB_SLM_MHCLITE_NPU_OPS", "1") != "1":
        return False
    if not mhclite_ops.npu_ops_available():
        return False
    try:
        return hasattr(torch.ops._C_ascend, "npu_hc_post")
    except AttributeError:
        return False


class MHCLite(nn.Module):
    """Lite hyper-connection operator (mcore MHCLite, permutation basis);
    math in float32, fused NPU ops when available, pure-torch fallback."""

    def __init__(self, config, is_head: bool = False):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        self.is_head = is_head
        self.num_perm_mats = math.factorial(self.hc_mult)
        self.mix_hc = self.hc_mult if is_head else 2 * self.hc_mult + self.num_perm_mats
        self.hc_fn = nn.Linear(
            self.hc_mult * config.hidden_size, self.mix_hc, bias=False, dtype=torch.float32
        )
        self.hc_gamma = nn.Parameter(
            torch.ones(self.hc_mult * config.hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_scale = nn.Parameter(
            torch.zeros(1 if is_head else 3, dtype=torch.float32), requires_grad=False
        )
        self.hc_base = nn.Parameter(
            torch.zeros(self.mix_hc, dtype=torch.float32), requires_grad=False
        )
        self._use_npu_ops = not is_head and _npu_mhclite_ops_available()

    def _perm_flat(self, device) -> torch.Tensor:
        e = self.hc_mult
        perms = torch.eye(e, device=device)[
            torch.tensor(list(permutations(range(e))), device=device)
        ]
        return perms.flatten(1)

    def _coefficients(self, x: torch.Tensor):
        # x: [t, e, d] -> h_pre [t, e] fp32, h_post [t, e] fp32, h_res [t, e, e] fp32
        e = self.hc_mult
        x_flat = x.flatten(1).float()
        # learned-gamma RMSNorm over the whole flattened stream vector
        x_norm = x_flat * torch.rsqrt(
            x_flat.square().mean(-1, keepdim=True) + self.norm_eps
        ) * self.hc_gamma
        logits = F.linear(x_norm, self.hc_fn.weight)
        hc_scale, hc_base = self.hc_scale.float(), self.hc_base.float()
        h_pre = torch.sigmoid(logits[:, :e] * hc_scale[0] + hc_base[:e])
        h_post = 2 * torch.sigmoid(logits[:, e : 2 * e] * hc_scale[1] + hc_base[e : 2 * e])
        coeffs = (logits[:, 2 * e :] * hc_scale[2] + hc_base[2 * e :]).softmax(-1)
        h_res = torch.matmul(coeffs, self._perm_flat(x.device)).view(-1, e, e)
        return h_pre, h_post, h_res

    def _torch_pre(self, x: torch.Tensor):
        # x: [t, e, d] -> y [t, d] in x.dtype; post/h_res stay fp32 for post()
        e = self.hc_mult
        h_pre, h_post, h_res = self._coefficients(x)
        y = torch.sum(h_pre.to(x.dtype).unsqueeze(-1) * x, dim=1)
        return y, h_post, h_res

    def pre(self, x: torch.Tensor):
        if self._use_npu_ops and x.dtype == torch.bfloat16:
            try:
                return mhclite_ops.lite_pre(
                    x, self.hc_fn.weight, self.hc_gamma,
                    self.hc_scale, self.hc_base, self.norm_eps,
                )
            except RuntimeError:
                self._use_npu_ops = False
        return self._torch_pre(x)

    def post(self, x, residual, post, comb):
        # x: [t, d]; residual: [t, e, d]; post: [t, e]; comb: [t, e_in, e_out]
        # out[o] = post[o] * x + sum_i comb[i, o] * residual[i]
        if self._use_npu_ops and x.dtype == torch.bfloat16:
            try:
                return torch.ops._C_ascend.npu_hc_post(
                    x.unsqueeze(0), residual.unsqueeze(0), post.unsqueeze(0),
                    comb.unsqueeze(0),
                ).squeeze(0)
            except RuntimeError:
                self._use_npu_ops = False
        y = post.unsqueeze(-1) * x.float().unsqueeze(-2)
        y = y + torch.matmul(comb.transpose(-1, -2), residual.float())
        return y.type_as(x)

    def head(self, x: torch.Tensor):
        # x: [t, e, d] -> y [t, d]; the op contract is the 32-lane layer
        # layout, so the head keeps the pure-torch path
        x_flat = x.flatten(1).float()
        x_norm = x_flat * torch.rsqrt(
            x_flat.square().mean(-1, keepdim=True) + self.norm_eps
        ) * self.hc_gamma
        logits = F.linear(x_norm, self.hc_fn.weight)
        h_pre = torch.sigmoid(logits * self.hc_scale[0] + self.hc_base).to(x.dtype)
        return torch.sum(h_pre.unsqueeze(-1) * x, dim=1)


class AILabSLMMHCLiteDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=100000)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )

        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = AILabSLMAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.mlp = AILabSLMMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_mhc = MHCLite(config)
        self.mlp_mhc = MHCLite(config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, post, comb = self.attn_mhc.pre(hidden_states)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        hidden_states = self.attn_mhc.post(hidden_states, residual, post, comb)

        residual = hidden_states
        hidden_states, post, comb = self.mlp_mhc.pre(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_mhc.post(hidden_states, residual, post, comb)
        return hidden_states


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class AILabSLMMHCLiteModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config.get_text_config()
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size
        self.scale_emb = config.scale_emb
        self.hc_mult = config.hc_mult

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: AILabSLMMHCLiteDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        def make_empty_intermediate_tensors(
            batch_size: int,
            dtype: torch.dtype,
            device: torch.device,
        ) -> IntermediateTensors:
            return IntermediateTensors(
                {
                    "hidden_states": torch.zeros(
                        (batch_size, config.hc_mult, config.hidden_size),
                        dtype=dtype,
                        device=device,
                    ),
                }
            )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors
        if get_pp_group().is_last_rank:
            self.hc_head = MHCLite(config, is_head=True)
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.hc_head = PPMissingLayer()
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            if self.scale_emb is not None:
                hidden_states = hidden_states * self.scale_emb
            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(positions, hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                }
            )

        hidden_states = self.hc_head.head(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name.endswith("scale"):
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class AILabSLMMHCLiteForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config.get_text_config()
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        self.model = AILabSLMMHCLiteModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
