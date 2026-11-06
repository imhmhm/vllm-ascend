// Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
/** torch binding for the standalone LitePreHeads Ascend C op.

The op binary lives outside CANN (built from the gitcode ops-transformer
tree); the aclnn symbols are resolved at runtime from libcust_opapi.so
via ASCEND_CUSTOM_OPP_PATH, which is exactly how the
cann_ops_transformer ACLNN_CMD loader prefers custom ops over the CANN
built-ins.
 */

#include <torch/extension.h>
#include <vector>

#include "aclnn_common.h"

std::vector<at::Tensor> lite_pre_heads(const at::Tensor &x, const at::Tensor &logits,
    const at::Tensor &scale, const at::Tensor &base, const at::Tensor &eps)
{
    const int64_t sb = x.size(0);
    const int64_t h = x.size(1) / 4;
    auto y = at::empty({sb, h}, x.options());
    auto hPre = at::empty({sb, 8}, scale.options());
    auto hPost = at::empty({sb, 8}, scale.options());
    auto coeff = at::empty({sb, 32}, scale.options());
    // the kernel always copies whole 8-token rows, so tail-block lanes past
    // sb need slack; the returned view hides it
    auto rstdPad = at::empty({(sb + 7) / 8 * 8}, scale.options());
    ACLNN_CMD(aclnnLitePreHeads, x, logits, scale, base, eps, y, hPre, hPost, coeff, rstdPad);
    return {y, hPre, hPost, coeff, rstdPad.narrow(0, 0, sb)};
}

// Scheme G: the whole scheme-E forward chain in one binding call.  The
// device kernels (W' build, logits GEMM, LitePreHeads, plus the small
// h_res GEMM) are identical to the Python chain; only the per-op Python /
// dispatch round-trips collapse into one, so the stream sees them
// back-to-back with kernel-to-kernel gaps only.  The raw bf16 logits ride
// along for the backward chain.
std::vector<at::Tensor> lite_pre_chain(const at::Tensor &x, const at::Tensor &w,
    const at::Tensor &gamma, const at::Tensor &scale, const at::Tensor &base,
    const at::Tensor &eps, const at::Tensor &permFlat)
{
    auto wp = at::mul(w, gamma.view({1, -1}));
    auto logitsRaw = at::matmul(x, wp.t());
    auto heads = lite_pre_heads(x, logitsRaw, scale, base, eps);
    auto coeff = heads[3];
    TORCH_CHECK(coeff.size(1) == 32, "res lanes expected at 8:32 of a 32-lane coeff");
    auto hRes = at::matmul(coeff.narrow(1, 8, 24), permFlat);
    return {heads[0], heads[1], heads[2], coeff, heads[4], hRes, logitsRaw};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("lite_pre_heads", &lite_pre_heads, "mhc_lite pre heads fused op (standalone Ascend C)");
    m.def("lite_pre_chain", &lite_pre_chain, "scheme-G one-call forward chain (W' + GEMM + heads + h_res)");
}
