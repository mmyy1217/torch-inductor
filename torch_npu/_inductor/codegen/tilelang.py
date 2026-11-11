"""Native TileLang template IR, candidate and Python-wrapper code generation.

TileLang programs are rendered during scheduling and compiled during wrapper loading.
"""

import hashlib
from pathlib import Path

import torch

from torch._inductor.ir import ChoiceCaller, TemplateBuffer, TensorBox
from torch._inductor.virtualized import V


@torch.compiler.assume_constant_result
def template_cache_key():
    """Snapshot template/toolchain versions into the FX graph when Dynamo traces it."""
    from ..runtime.tilelang import compiler_fingerprint

    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for name in (
        "kernel/flex_attention.py",
        "kernel/tilelang_flex_attention.py",
        "kernel/tilelang_flex_kernel.py",
        "codegen/tilelang.py",
        "codegen/tilelang_template.py",
        "runtime/tilelang.py",
    ):
        digest.update((root / name).read_bytes())
    digest.update(compiler_fingerprint().encode())
    return digest.hexdigest()


class TileLangTemplateKernel:
    def __init__(self, spec):
        self.spec = spec

    def render(self):
        from .tilelang_template import render_flex_attention

        return render_flex_attention(self.spec)


class TileLangTemplateBuffer(TemplateBuffer):
    def __init__(self, layout, inputs, spec):
        def make_kernel_render(template_node, *args, **kwargs):
            kernel = TileLangTemplateKernel(template_node.spec)
            return kernel, kernel.render

        super().__init__(layout, inputs, make_kernel_render=make_kernel_render)
        self.spec = spec
        self.allow_prologue_fusion = False
        self.allow_epilogue_fusion = False

    def __str__(self):
        return f"TileLangTemplateBuffer(layout={self.layout})"


class TileLangTemplateCaller(ChoiceCaller):
    def __init__(self, inputs, layout, spec):
        super().__init__(
            "tilelang_flex_attention",
            inputs,
            layout,
            "Ascend TileLang static BF16 Cube FlexAttention",
        )
        self.spec = spec

    def output_node(self):
        return TensorBox.create(
            TileLangTemplateBuffer(self.layout, self.input_nodes, self.spec)
        )

    def call_name(self):
        return "torch_npu._inductor.runtime.tilelang.PreparedTileLangKernel.launch"

    def to_callable(self):
        raise NotImplementedError(
            "TileLang singleton candidates compile through template rendering"
        )

    def hash_key(self):
        return hashlib.sha256(
            (template_cache_key() + str(self.layout) + self.spec).encode()
        ).hexdigest()

    def benchmark(self, *args, out):
        # Future mixed-candidate selection needs an explicit common timing policy.
        raise NotImplementedError(
            "TileLang v1 supports singleton selection, not autotune"
        )


def is_tilelang_template(node):
    return node.is_template() and isinstance(
        node.get_template_node(), TileLangTemplateBuffer
    )


def contains_tilelang_template(node):
    return any(is_tilelang_template(n) for n in node.get_nodes())


def codegen_tilelang_template(
    scheduling, template_node, epilogue_nodes=(), prologue_nodes=()
):
    if epilogue_nodes or prologue_nodes:
        raise NotImplementedError("TileLang v1 does not support template fusion")
    if V.graph.cpp_wrapper or V.graph.aot_mode:
        raise NotImplementedError("TileLang v1 supports the Python wrapper only")
    buf = template_node.node
    if not isinstance(buf, TileLangTemplateBuffer):
        raise TypeError(type(buf))
    wrapper = V.graph.wrapper_code
    wrapper.imports.writeline(
        "from torch_npu._inductor.runtime.tilelang import compile_tilelang"
    )
    kernel, render = buf.make_kernel_render(buf)
    source = render()
    device_index = buf.get_device().index
    if device_index is None:
        raise NotImplementedError("TileLang requires an explicit NPU device index")
    kernel_cache_key = f"tilelang:{device_index}:{source}"
    if kernel_cache_key in wrapper.src_to_kernel:
        kernel_name = wrapper.src_to_kernel[kernel_cache_key]
    else:
        kernel_name = f"tilelang_flex_{wrapper.next_kernel_suffix()}"
        wrapper.src_to_kernel[kernel_cache_key] = kernel_name
        wrapper.define_kernel(
            kernel_name,
            f"compile_tilelang('''\n{source}''', device_index={device_index})",
            metadata="# TileLang static FlexAttention template; compiled before call()",
        )
    # mark_run allocates the output through Inductor's normal memory planner.
    template_node.mark_run()
    inputs = [x.codegen_reference() for x in buf.inputs]
    wrapper.writeline(
        f"{kernel_name}.launch({', '.join(inputs)}, out={buf.get_name()})"
    )
    scheduling.scheduler.free_buffers()
