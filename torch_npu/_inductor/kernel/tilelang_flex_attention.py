"""Lower default dense FlexAttention to a static BF16 TileLang template."""

import json
import math
import operator

import sympy
import torch
from torch._inductor.ir import FixedLayout, TemplateBuffer
from torch._inductor.select_algorithm import autotune_select_algorithm
from torch._inductor.virtualized import V


def reject(message):
    raise NotImplementedError("TileLang FlexAttention v2: " + message)


def check_device(device):
    if device.type != "npu":
        reject("requires Ascend NPU")


def static_shape(x):
    shape = x.get_size()
    if any(not isinstance(d, (int, sympy.Integer)) or int(d) <= 0 for d in shape):
        reject("positive static dimensions required")
    return tuple(map(int, shape))


def contiguous(x):
    stride = 1
    for size, actual in zip(reversed(static_shape(x)), reversed(x.get_stride())):
        if size != 1 and actual != stride:
            return False
        stride *= size
    return True


def validate(
    query, key, value, score, block_mask, scale, options, score_buffers, mask_buffers
):
    # Set by the patched Python entry before PyTorch constructs its default BlockMask.
    if not options.get("_NPU_DENSE_ATTENTION"):
        reject(
            "dense only: omit score_mod/block_mask; import torch_npu._inductor before binding flex_attention"
        )
    if not V.graph.is_inference or V.graph.is_backward:
        reject("inference-only; backward is not implemented")
    if V.graph.cpp_wrapper or V.graph.aot_mode:
        reject("Python wrapper required")
    for x in (query, key, value):
        check_device(x.get_device())
        if (
            len(static_shape(x)) != 4
            or not contiguous(x)
            or x.get_dtype() != torch.bfloat16
        ):
            reject("contiguous static BF16 rank-4 Q/K/V required")
    b, h, m, d = static_shape(query)
    bk, hk, n, dk = static_shape(key)
    if (
        static_shape(value) != (bk, hk, n, d)
        or bk != b
        or h % hk != 0
        or d not in (128, 256, 512)
        or dk != d
    ):
        reject(
            "matching batch and D in {128, 256, 512} required; Q heads must be divisible by KV heads"
        )
    if max(b * h * m * d, b * hk * n * d) >= 2**31:
        reject("Q/K/V element counts must be below 2**31")
    if m % 128 or n % 128:
        reject("Q/KV sequence lengths must be multiples of 128")
    if not query.get_device() == key.get_device() == value.get_device():
        reject("Q/K/V device mismatch")
    if score_buffers or mask_buffers:
        reject("captured tensors are not supported")
    if not isinstance(scale, (float, int)) or not math.isfinite(scale):
        reject("finite static scale required")
    # PyTorch requests internal LSE in grad mode even for inputs without gradients.
    # Inference and output-consumer guards keep actual auxiliary outputs unsupported.
    if options.get("_NPU_EXPLICIT_AUX", True) or options.get("OUTPUT_MAX"):
        reject("auxiliary outputs are not supported")
    allowed = {
        "BACKEND",
        "PRESCALE_QK",
        "ROWS_GUARANTEED_SAFE",
        "BLOCKS_ARE_CONTIGUOUS",
        "WRITE_DQ",
        "OUTPUT_LOGSUMEXP",
        "OUTPUT_MAX",
        "_NPU_EXPLICIT_SCORE_MOD",
        "_NPU_DENSE_ATTENTION",
        "_NPU_EXPLICIT_AUX",
        "_NPU_TILELANG_CACHE_KEY",
    }
    if (
        set(options) - allowed
        or options.get("PRESCALE_QK")
        or options.get("BACKEND", "AUTO") != "AUTO"
    ):
        reject("unsupported kernel_options")
    node = V.graph.current_node
    if not node.users or any(
        u.op != "call_function"
        or u.target is not operator.getitem
        or u.args != (node, 0)
        for u in node.users
    ):
        reject("only the attention tensor may be consumed")
    return dict(shape=(b, h, m, n, d, hk), scale=float(scale))


def lower_tilelang_flex_attention(
    query,
    key,
    value,
    score,
    block_mask,
    scale,
    kernel_options,
    score_buffers,
    mask_buffers,
):
    inputs = [TemplateBuffer.realize_template_input(x) for x in (query, key, value)]
    spec = validate(
        *inputs, score, block_mask, scale, kernel_options, score_buffers, mask_buffers
    )
    from torch_npu._inductor.codegen.tilelang import TileLangTemplateCaller

    layout = FixedLayout(
        query.get_device(), query.get_dtype(), query.get_size(), query.get_stride()
    )
    choice = TileLangTemplateCaller(inputs, layout, json.dumps(spec, sort_keys=True))
    out, _ = autotune_select_algorithm(
        "tilelang_flex_attention", [choice], inputs, layout, return_multi_template=False
    )
    return out, None, None
