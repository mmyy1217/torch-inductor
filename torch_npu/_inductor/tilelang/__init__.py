"""Initialization entry point for the TileLang Inductor backend."""


def initialize_tilelang_backend() -> None:
    """Register TileLang as the NPU Inductor scheduling backend."""
    from torch._inductor import lowering as inductor_lowering
    from torch._inductor.codegen.common import register_backend_for_device

    from torch_npu._inductor.codegen.cpp_wrapper_npu import CppWrapperNpu
    from torch_npu._inductor.codegen.ir import patch_indexing, patch_loop_body
    from torch_npu._inductor.codegen.wrapper import NPUPythonWrapperCodeGen
    from torch_npu._inductor.decomposition import (
        _register_triton_decompositions,
    )
    from torch_npu._inductor.kernel import (
        _register_npu_inductor_addmm,
        _register_npu_inductor_bmm,
        _register_npu_inductor_mm,
    )
    from torch_npu._inductor.lowering import (
        _register_npu_inductor_fallbacks,
        npu_make_fallback,
    )
    from torch_npu._inductor.scheduler import patch_scheduler
    from torch_npu._inductor.utils import disable_foreach

    from .codegen.tilelang_scheduling import TileLangScheduling

    # TileLang uses NPUIndexTritonKernel's range-tree/index transformation.
    # Install the same LoopBody helpers as the Triton backend so a TileLang-only
    # process does not depend on another backend having initialized first.
    patch_loop_body()
    patch_indexing()

    register_backend_for_device(
        "npu", TileLangScheduling, NPUPythonWrapperCodeGen, CppWrapperNpu
    )

    inductor_lowering.make_fallback = npu_make_fallback
    _register_triton_decompositions()
    _register_npu_inductor_fallbacks()
    _register_npu_inductor_mm()
    _register_npu_inductor_addmm()
    _register_npu_inductor_bmm()

    patch_scheduler()
    disable_foreach()


__all__ = ["initialize_tilelang_backend"]
