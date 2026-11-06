# Copyright 2026 The vLLM-Ascend team.
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

"""mhc-lite (permutation-basis hyper-connections) pre operator.

Thin wrapper over the LitePreHeads / lite_pre_chain Ascend C op: the whole
hc_pre chain (learned-gamma RMSNorm folding, the coefficient GEMM and the
h_pre/h_post/h_res heads) runs as one extension call. The op binary is
built outside CANN from the ops-transformer tree and resolves at runtime
through ASCEND_CUSTOM_OPP_PATH, which must be set before torch_npu is
imported; without it the caller falls back to the pure-torch path.

The wrapper is registered as a torch custom op so fullgraph torch.compile
treats it as opaque: the lane-index / eps / permutation constants are
built lazily on first (eager) execution and then cached, because a
host-to-device copy baked into the compiled graph makes ACL graph
capture fail.
"""

import os
from itertools import permutations
from pathlib import Path

import torch

# per-logit-lane scale expansion: s0 x4 | s1 x4 | s2 x24 (e = 4, e! = 24)
_LANE_IDX = [0] * 4 + [1] * 4 + [2] * 24

_EXT = None
_PERM_FLAT = {}
_LANE_IDX_T = {}
_EPS8 = {}


def _ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    if not os.environ.get("ASCEND_CUSTOM_OPP_PATH"):
        raise RuntimeError(
            "mhc-lite NPU ops need ASCEND_CUSTOM_OPP_PATH pointing at the "
            "vendor tree built by sync_and_build.sh (set it before importing "
            "torch_npu)")
    import torch_npu
    from torch.utils.cpp_extension import load

    torch_npu_dir = Path(torch_npu.__file__).parent
    cann_inc = Path(
        os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.1.1")
    ) / "python/site-packages/cann_ops_transformer/common/inc"
    _EXT = load(
        name="vllm_ascend_lite_pre_ext",
        sources=[str(Path(__file__).parent / "op_src" / "lite_pre.cpp")],
        extra_include_paths=[
            str(torch_npu_dir / "include"),
            str(torch_npu_dir / "include/third_party/acl/inc"),
            str(cann_inc),
        ],
        extra_ldflags=[f"-L{torch_npu_dir}/lib", "-ltorch_npu"],
        verbose=False,
    )
    return _EXT


def _perm_flat(device, hc_mult: int) -> torch.Tensor:
    # row-major flattened permutation matrices [e!, e*e]; h_res = coeff @ perm
    if (device, hc_mult) not in _PERM_FLAT:
        e = hc_mult
        perms = torch.eye(e, device=device)[
            torch.tensor(list(permutations(range(e))), device=device)
        ]
        _PERM_FLAT[(device, hc_mult)] = perms.flatten(1)
    return _PERM_FLAT[(device, hc_mult)]


def _lane_idx_t(device) -> torch.Tensor:
    if device not in _LANE_IDX_T:
        _LANE_IDX_T[device] = torch.tensor(_LANE_IDX, device=device)
    return _LANE_IDX_T[device]


def _eps8_t(device, eps: float) -> torch.Tensor:
    if (device, eps) not in _EPS8:
        _EPS8[(device, eps)] = torch.full(
            (8,), eps, dtype=torch.float32, device=device
        )
    return _EPS8[(device, eps)]


def npu_ops_available() -> bool:
    try:
        _ext()
        return True
    except Exception:
        return False


@torch.library.custom_op("vllm_ascend_mhclite::lite_pre", mutates_args=())
def _lite_pre(
    x: torch.Tensor,
    weight: torch.Tensor,
    gamma: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t, e, d = x.shape
    y, _hpre, hpost8, _coeff, _rstd, h_res, _logits = _ext().lite_pre_chain(
        x.reshape(t, e * d),
        weight.type_as(x),
        gamma.type_as(x),
        scale.float()[_lane_idx_t(x.device)],
        base.float(),
        _eps8_t(x.device, norm_eps),
        _perm_flat(x.device, e),
    )
    # hpost8 carries [pre | post] lanes; npu_hc_post rejects non-contiguous
    # inputs, so the slice materializes here
    return y, hpost8[:, e:].contiguous(), h_res.view(t, e, e)


@_lite_pre.register_fake
def _(
    x, weight, gamma, scale, base, norm_eps,
):
    t, e, d = x.shape
    return (
        x.new_empty((t, d)),
        scale.new_empty((t, e)),
        scale.new_empty((t, e, e)),
    )


def lite_pre(
    x: torch.Tensor,
    weight: torch.Tensor,
    gamma: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """x [t, e, d] bf16 -> (y [t, d], h_post [t, e], h_res [t, e, e]).

    weight [mix, e*d] and gamma [e*d] arrive in fp32 and cast to the stream
    dtype for the op; h_post / h_res stay fp32 for the matching post().
    """
    return torch.ops.vllm_ascend_mhclite.lite_pre(x, weight, gamma, scale, base, norm_eps)
