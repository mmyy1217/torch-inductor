"""TileLang scheduling integration for the NPU Inductor backend."""

from __future__ import annotations

from typing import Any, Optional, cast

import sympy
import torch
from torch.utils._ordered_set import OrderedSet

from torch._inductor import config
from torch._inductor import scheduler as inductor_scheduler
from torch._inductor.codegen.common import BackendFeature, IndentedBuffer, TensorArg
from torch._inductor.codegen.simd import SIMDKernel, schedule_log
from torch._inductor.codegen.triton import get_fused_kernel_name, get_kernel_metadata
from torch._inductor.utils import Placeholder, sympy_product
from torch._inductor.virtualized import V

from torch_npu._inductor.codegen.npu_kernel_features import NPUKernelFeatures
from torch_npu._inductor.codegen.scheduling import NPUTritonScheduling
from torch_npu._inductor.codegen.triton_utils import get_byte_per_numel

from .tilelang_kernel import (
    TileLangKernel,
    _DEFAULT_REDUCTION_RBLOCK,
    _DEFAULT_REDUCTION_XBLOCK,
    _DEFAULT_XBLOCK,
    _MAX_REDUCTIONS_PER_KERNEL,
)


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

class TileLangScheduling(NPUTritonScheduling):
    """
    Inductor scheduling backend that emits TileLang kernels for Ascend NPU.

    Registered via ``register_backend_for_device`` in __init__.py when a
    compile invocation selects ``options={"npu_backend": "tilelang"}``.
    """

    kernel_type: type[Any] = TileLangKernel

    backend_features: OrderedSet[BackendFeature] = OrderedSet()

    def __init__(self, input_scheduler) -> None:
        super().__init__(input_scheduler)
        self.kernel_type = TileLangKernel
        self._triton_scheduling = NPUTritonScheduling(input_scheduler)

    @classmethod
    def get_backend_features(cls, device: torch.device) -> OrderedSet[BackendFeature]:
        return cls.backend_features

    def codegen_comment(self, node_schedule) -> None:
        wrapper = V.graph.wrapper_code
        origins, _ = get_kernel_metadata(node_schedule, wrapper)
        if origins:
            wrapper.writeline(origins)

    def codegen_sync(self) -> None:
        V.graph.wrapper_code.writeline("torch.npu.synchronize()")

    def make_ttir_for_check(self, src_code: str) -> None:
        # The parent hook parses Triton-specific ``inductor_meta`` and asks
        # Triton to build TTIR.  TileLang source is validated later by
        # tilelang.compile(target="ascend") instead.
        return None

    def codegen_template(
        self,
        template_node,
        epilogue_nodes: list,
        only_gen_src_code: bool = False,
    ):
        return self._triton_scheduling.codegen_template(
            template_node, epilogue_nodes, only_gen_src_code
        )

    def _fallback_to_triton(self, node, reason: str):
        schedule_log.debug("TileLang fallback to NPU Triton: %s", reason)
        return self._triton_scheduling.codegen_node(node)

    def _unsupported_tilelang_reason(
        self,
        nodes: list[inductor_scheduler.SchedulerNode],
        reduction_numel: sympy.Expr,
    ) -> Optional[str]:
        reduction_nodes = [node for node in nodes if node.is_reduction()]
        if reduction_nodes:
            return "A5 reduction VF migration is pending"
        if len(reduction_nodes) > _MAX_REDUCTIONS_PER_KERNEL:
            names = ", ".join(node.get_name() for node in reduction_nodes)
            return (
                "multiple reductions in one TileLang kernel are not yet "
                f"supported ({len(reduction_nodes)} reductions: {names})"
            )

        for node in nodes:
            if node.is_template():
                return f"{node.get_name()} is a template node"
            if node.is_split_scan():
                return f"{node.get_name()} is a split-scan node"

        return None

    @staticmethod
    def _numel_product(value):
        if isinstance(value, (list, tuple, sympy.Tuple)):
            return sympy_product(value)
        return value

    @classmethod
    def _same_numel(cls, lhs, rhs) -> bool:
        lhs = cls._numel_product(lhs)
        rhs = cls._numel_product(rhs)
        try:
            return bool(sympy.simplify(lhs - rhs) == 0)
        except Exception:
            return lhs == rhs

    @staticmethod
    def _reduction_types_for_node(node) -> list[str]:
        reduction_types: list[str] = []
        for scheduler_node in node.get_nodes():
            ir_node = getattr(scheduler_node, "node", None)
            data = getattr(ir_node, "data", None)
            reduction_type = getattr(data, "reduction_type", None)
            if reduction_type is not None:
                reduction_types.append(str(reduction_type))
        return reduction_types

    @classmethod
    def _is_supported_reduction_fusion(cls, node1, node2) -> bool:
        reduction_nodes = sorted(
            [*node1.get_nodes(), *node2.get_nodes()],
            key=lambda node: getattr(node, "min_order", 0),
        )
        reduction_types: list[str] = []
        for scheduler_node in reduction_nodes:
            ir_node = getattr(scheduler_node, "node", None)
            data = getattr(ir_node, "data", None)
            reduction_type = getattr(data, "reduction_type", None)
            if reduction_type is not None:
                reduction_types.append(str(reduction_type))
        return reduction_types in (["max", "sum"], ["sum", "sum"])

    def can_fuse(self, node1, node2):
        if node1.is_reduction() and node2.is_reduction():
            if not self._is_supported_reduction_fusion(node1, node2):
                return False
            return super().can_fuse(node1, node2)

        if node1.is_reduction() != node2.is_reduction():
            reduction_node = node1 if node1.is_reduction() else node2
            pointwise_node = node2 if node1.is_reduction() else node1
            reduction_types = self._reduction_types_for_node(reduction_node)
            if len(reduction_types) > 1 and reduction_types not in (
                ["max", "sum"],
                ["sum", "sum"],
            ):
                return False
            _, (pointwise_numel, pointwise_rnumel) = pointwise_node.group
            _, (reduction_numel, reduction_rnumel) = reduction_node.group
            is_vector_epilogue = self._same_numel(
                pointwise_numel,
                self._numel_product(reduction_numel)
                * self._numel_product(reduction_rnumel),
            )
            if is_vector_epilogue and reduction_types not in (
                ["max", "sum"],
                ["sum", "sum"],
                ["sum"],
            ):
                return False

            if self._same_numel(pointwise_rnumel, 1) and (
                self._same_numel(pointwise_numel, reduction_numel)
                or is_vector_epilogue
            ):
                try:
                    return all(
                        SIMDKernel.is_compatible(
                            (reduction_numel, reduction_rnumel), n.get_ranges()
                        )
                        for n in pointwise_node.get_nodes()
                    )
                except Exception:
                    return True

        return super().can_fuse(node1, node2)

    can_fuse_vertical = can_fuse
    can_fuse_horizontal = can_fuse

    def codegen_node(
        self,
        node: inductor_scheduler.FusedSchedulerNode | inductor_scheduler.SchedulerNode,
    ):
        nodes = cast(list[inductor_scheduler.SchedulerNode], node.get_nodes())
        _, (numel, rnumel) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group

        reason = self._unsupported_tilelang_reason(nodes, rnumel)
        if reason is not None:
            return self._fallback_to_triton(node, reason)

        try:
            # Keep the per-axis numel values while constructing the schedule,
            # but use scalar products for feature queries.  SIMDKernelFeatures
            # simplifies numel in its constructor and cannot accept a SymPy
            # Tuple directly.
            node_schedule = self.generate_node_schedule(nodes, numel, rnumel)
            schedule_log.debug("Schedule:\n %s", node_schedule)
            kernel_features = NPUKernelFeatures(
                node_schedule,
                self._numel_product(numel),
                self._numel_product(rnumel),
            )
            return self.codegen_node_schedule(kernel_features, nodes, node)
        except NotImplementedError as exc:
            return self._fallback_to_triton(node, str(exc))

    def codegen_node_schedule(self, kernel_features, nodes=None, origin_node=None):
        if nodes is None:
            nodes = list(kernel_features.scheduler_nodes())
        return super().codegen_node_schedule(kernel_features, nodes, origin_node)

    @classmethod
    def select_tiling(cls, nodes, numel, reduction_numel=1):
        if isinstance(numel, (list, tuple, sympy.Tuple)):
            numel = sympy_product(numel)
        if isinstance(reduction_numel, (list, tuple, sympy.Tuple)):
            reduction_numel = sympy_product(reduction_numel)
        if str(reduction_numel) == "1":
            # Keep the empty reduction range tree used by the default Triton
            # scheduling path.  Shared NPU indexing analysis treats the last
            # range tree as the reduction tree, even for pointwise kernels.
            return {"x": numel, "r": sympy.S.One}
        return {"x": numel, "r": reduction_numel}

    def define_kernel(
        self,
        src_code: str,
        node_schedule,
        kernel: TileLangKernel,
        traced_graph_hash: Optional[str] = None,
        kernel_graph_hash: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        Splice a shape-keyed caching wrapper into ``wrapper.header``.

        Pattern emitted at module level::

            import tilelang as _tilelang_<N>

            def _prim_factory_<name>(xnumel):
                # @T.prim_func definition (src_code)
                return <name>_prim_fn

            _<name>_cache = {}

            def <name>(in_ptr0, ..., xnumel):
                _key = (int(xnumel),)
                if _key not in _<name>_cache:
                    _<name>_cache[_key] = _tilelang_<N>.compile(
                        _prim_factory_<name>(_key[0]), target='ascend', execution_backend='tvm_ffi')
                _<name>_cache[_key](in_ptr0, ...)
        """
        wrapper = V.graph.wrapper_code

        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
            return kernel_name, src_code.replace(str(Placeholder.KERNEL_NAME), kernel_name)

        fused_name = (
            get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
            if config.triton.descriptive_names else ""
        )
        suffix      = wrapper.next_kernel_suffix()
        kernel_name = "_".join(filter(None, ["tilelang", fused_name, suffix]))
        wrapper.src_to_kernel[src_code] = kernel_name

        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), kernel_name)

        _, call_args, signature, _ = kernel.args.python_argdefs()
        tensor_call_args = [a for a, s in zip(call_args, signature) if isinstance(s, TensorArg)]
        output_arg_names = {
            var
            for _, (var, _, _) in kernel._output_buffer_items()
        }
        output_arg_indices = [
            i for i, arg in enumerate(tensor_call_args) if arg in output_arg_names
        ]
        if not output_arg_indices and tensor_call_args:
            output_arg_indices = [-1]
        normalized_output_arg_indices = {
            i if i >= 0 else len(tensor_call_args) + i
            for i in output_arg_indices
        }
        profile_input_args = [
            arg
            for i, arg in enumerate(tensor_call_args)
            if i not in normalized_output_arg_indices
        ]
        tensor_dtypes = [
            s.dtype for s in signature if isinstance(s, TensorArg)
        ]
        try:
            dtype_bytes = max(get_byte_per_numel(dtype) for dtype in tensor_dtypes)
        except Exception:
            dtype_bytes = 4
        buffer_count = min(
            max(
                len(kernel._input_items())
                + len(kernel._output_buffer_items())
                + len(kernel._reduction_outputs),
                1,
            ),
            3,
        )
        is_reduction_kernel = bool(kernel._reduction_outputs)
        fallback_xblock_hint = (
            _DEFAULT_REDUCTION_XBLOCK if is_reduction_kernel else _DEFAULT_XBLOCK
        )
        active_trees     = kernel.active_range_trees()
        numel_arg_names  = [f"{t.prefix}numel" for t in active_trees]
        static_axis_args = kernel.tilelang_static_axis_args()
        static_axis_arg_names = [name.lstrip("_") for name, _ in static_axis_args]
        shape_factory_arg_names = numel_arg_names + static_axis_arg_names
        outer_arg_list = (
            tensor_call_args + numel_arg_names + static_axis_arg_names
        )
        tilelang_sorted_axis = list(getattr(kernel, "sorted_axis", []) or [])
        tilelang_pointwise_axis_lengths = (
            getattr(kernel, "_tl_pointwise_axis_lengths", None)
            if not is_reduction_kernel
            else None
        )
        tilelang_pointwise_axis_names = (
            getattr(kernel, "_tl_pointwise_axis_names", None)
            if tilelang_pointwise_axis_lengths
            else None
        )
        tilelang_axis_names = [axis.name for axis in tilelang_sorted_axis]
        tilelang_tiling_axis = [
            axis.sorted_order for axis in getattr(kernel, "tiling_axis", [])
            if axis.sorted_order is not None
        ]
        tilelang_no_loop_axis = [
            axis.sorted_order for axis in getattr(kernel, "tiling_axis", [])
            if axis.sorted_order is not None and getattr(axis, "is_no_loop_axis", False)
        ]
        tilelang_split_axis = [
            axis.sorted_order for axis in getattr(kernel, "split_axis", [])
            if axis.sorted_order is not None
        ]
        tilelang_low_dims = sorted(int(dim) for dim in getattr(kernel, "low_dims", set()))
        if tilelang_pointwise_axis_lengths and tilelang_pointwise_axis_names:
            tilelang_axis_names = list(tilelang_pointwise_axis_names)
            tilelang_tiling_axis = list(range(len(tilelang_axis_names)))
            tilelang_no_loop_axis = []
            tilelang_split_axis = list(range(len(tilelang_axis_names)))
            tilelang_low_dims = []
        ranked_pointwise = bool(
            tilelang_pointwise_axis_lengths
            and tilelang_pointwise_axis_names
            and len(tilelang_pointwise_axis_names) > 1
        )
        ranked_reduction = bool(
            is_reduction_kernel
            and sum(
                not axis_name.startswith("r")
                for axis_name in tilelang_axis_names
            ) > 1
            and sum(
                axis_name.startswith("r")
                for axis_name in tilelang_axis_names
            ) == 1
        )
        ranked_tiling = ranked_pointwise or ranked_reduction
        if ranked_tiling:
            tiling_arg_names = [
                f"{axis_name.upper()}BLOCK"
                for axis_name in tilelang_axis_names
            ]
        else:
            tiling_arg_names = ["X0BLOCK", "X0BLOCK_SUB"]
            if is_reduction_kernel:
                tiling_arg_names.append("R0BLOCK")
        factory_arg_names = shape_factory_arg_names + tiling_arg_names
        tilelang_persistent_reduction = bool(getattr(kernel, "persistent_reduction", False))
        try:
            tilelang_dual_reduction = bool(kernel.numof_reduction_axis() > 1)
        except Exception:
            tilelang_dual_reduction = False

        origins, detailed = get_kernel_metadata(node_schedule, wrapper)
        meta_comment = f"{origins}\n{detailed}".strip()

        tl_pkg_root: Optional[str] = None
        try:
            import tilelang as _tl
            import os as _os
            tl_pkg_root = _os.path.dirname(_os.path.dirname(_tl.__file__))
        except ImportError:
            pass

        import_alias = f"_tilelang_{suffix}"
        cache_var    = f"_{kernel_name}_cache"
        factory_fn   = f"_prim_factory_{kernel_name}"
        prim_fn_name = f"{kernel_name}_prim_fn"

        code = IndentedBuffer()
        code.writeline(f"\n# TileLang kernel - {meta_comment}")
        if tl_pkg_root:
            code.writeline("import sys as _sys")
            code.writeline(
                f"if {tl_pkg_root!r} not in _sys.path: "
                f"_sys.path.insert(0, {tl_pkg_root!r})"
            )
        code.writeline(f"import tilelang as {import_alias}")
        autotune_enabled_fn = f"_tilelang_autotune_enabled_{suffix}"
        autotune_warmup_fn = f"_tilelang_autotune_warmup_{suffix}"
        autotune_rep_fn = f"_tilelang_autotune_rep_{suffix}"
        autotune_timeout_fn = f"_tilelang_autotune_timeout_{suffix}"
        xblock_configs_fn = f"_tilelang_xblock_configs_{suffix}"
        ascend_pass_configs_fn = f"_tilelang_ascend_pass_configs_{suffix}"
        code.writeline(
            "from torch_npu._inductor.tilelang.codegen."
            "tilelang_kernel import "
            f"_tilelang_autotune_enabled as {autotune_enabled_fn}, "
            f"_tilelang_autotune_warmup as {autotune_warmup_fn}, "
            f"_tilelang_autotune_rep as {autotune_rep_fn}, "
            f"_tilelang_autotune_timeout as {autotune_timeout_fn}, "
            f"_tilelang_xblock_configs as {xblock_configs_fn}, "
            f"_tilelang_ascend_pass_configs as {ascend_pass_configs_fn}"
        )
        code.writeline("")

        factory_params = (
            ", ".join(factory_arg_names) if factory_arg_names else "_dummy=None"
        )
        code.writeline(f"def {factory_fn}({factory_params}):")
        with code.indent():
            if numel_arg_names:
                for arg_name in numel_arg_names:
                    code.writeline(f"_{arg_name} = {arg_name}")
                    if arg_name.startswith("r"):
                        code.writeline(f"_rnumel = {arg_name}")
            else:
                code.writeline("_xnumel = 1")
            for var_name, _extent in static_axis_args:
                code.writeline(f"{var_name} = {var_name.lstrip('_')}")
            code.splice(src_code)
            code.writeline(f"return {prim_fn_name}")

        code.writeline("")
        code.writeline(f"{cache_var} = {{}}")
        code.writeline("")

        code.writeline(f"def {kernel_name}({', '.join(outer_arg_list)}):")
        with code.indent():
            if numel_arg_names:
                key_items = [f"int({n})" for n in numel_arg_names]
                key_items.extend(
                    f"int({name})" for name in static_axis_arg_names
                )
                code.writeline(
                    f"_key = ({', '.join(key_items)},)"
                )
            else:
                if static_axis_args:
                    code.writeline(
                        f"_key = ({', '.join(f'int({name})' for name in static_axis_arg_names)},)"
                    )
                else:
                    code.writeline("_key = ('static',)")
            code.writeline(f"if _key not in {cache_var}:")
            with code.indent():
                shape_factory_call = (
                    ", ".join(f"_key[{i}]" for i in range(len(shape_factory_arg_names)))
                    if shape_factory_arg_names else ""
                )
                xnumel_expr = "int(xnumel)" if "xnumel" in numel_arg_names else "1"
                rnumel_expr = "int(rnumel)" if "rnumel" in numel_arg_names else "1"
                axis_numel_exprs: list[str] = []
                def axis_numel_expr_from_length(length: sympy.Expr) -> str:
                    simplified_length = V.graph.sizevars.simplify(length)
                    if isinstance(simplified_length, (sympy.Integer, int)):
                        return str(int(simplified_length))
                    for tree in active_trees:
                        prefix_arg_name = f"{tree.prefix}numel"
                        if (
                            prefix_arg_name in numel_arg_names
                            and V.graph.sizevars.statically_known_equals(
                                simplified_length,
                                tree.numel,
                            )
                        ):
                            return f"int({prefix_arg_name})"
                    for static_index, (_name, extent) in enumerate(
                        static_axis_args
                    ):
                        if V.graph.sizevars.statically_known_equals(
                            simplified_length,
                            extent,
                        ):
                            key_index = len(numel_arg_names) + static_index
                            return f"int(_key[{key_index}])"
                    return xnumel_expr

                if tilelang_pointwise_axis_lengths:
                    axis_numel_exprs = [
                        axis_numel_expr_from_length(length)
                        for length in tilelang_pointwise_axis_lengths
                    ]
                else:
                    for axis in tilelang_sorted_axis:
                        if axis.prefix == "r":
                            axis_numel_exprs.append(rnumel_expr)
                        else:
                            prefix_arg_name = f"{axis.prefix}numel"
                            simplified_length = V.graph.sizevars.simplify(axis.length)
                            if isinstance(simplified_length, (sympy.Integer, int)):
                                axis_numel_exprs.append(str(int(simplified_length)))
                            elif prefix_arg_name in numel_arg_names:
                                axis_numel_exprs.append(f"int({prefix_arg_name})")
                            else:
                                axis_numel_exprs.append(xnumel_expr)
                if not axis_numel_exprs:
                    axis_numel_exprs = [xnumel_expr, rnumel_expr] if is_reduction_kernel else [xnumel_expr]
                axis_numels_expr = f"[{', '.join(axis_numel_exprs)}]"
                code.writeline("_compiled_kernel = None")
                code.writeline(
                    f"_tilelang_configs = {xblock_configs_fn}("
                    f"{xnumel_expr}, {rnumel_expr}, "
                    f"is_reduction={is_reduction_kernel!r}, "
                    f"dtype_bytes={dtype_bytes!r}, "
                    f"buffer_count={buffer_count!r}, "
                    f"axis_numels={axis_numels_expr}, "
                    f"axis_names={tilelang_axis_names!r}, "
                    f"tiling_axis={tilelang_tiling_axis!r}, "
                    f"no_loop_axis={tilelang_no_loop_axis!r}, "
                    f"split_axis={tilelang_split_axis!r}, "
                    f"low_dims={tilelang_low_dims!r}, "
                    f"persistent_reduction={tilelang_persistent_reduction!r}, "
                    f"dual_reduction={tilelang_dual_reduction!r})"
                )
                # A5's first path fixes one version per materialized buffer.
                # Do not reinterpret the legacy NPUIR boolean as a version plan.
                code.writeline(
                    "_tilelang_configs = [c for c in _tilelang_configs "
                    "if not c.get('multibuffer', False)]"
                )
                fallback_tiling_vars: list[str] = []
                if ranked_tiling:
                    for axis_index, tiling_arg_name in enumerate(tiling_arg_names):
                        fallback_var = (
                            f"_tilelang_fallback_{tiling_arg_name.lower()}"
                        )
                        fallback_tiling_vars.append(fallback_var)
                        default_expr = (
                            f"max(1, min(int({axis_numel_exprs[axis_index]}), 2))"
                        )
                        code.writeline(
                            f"{fallback_var} = ("
                            f"_tilelang_configs[0].get({tiling_arg_name!r}, "
                            f"{default_expr}) if _tilelang_configs else "
                            f"{default_expr})"
                        )
                else:
                    fallback_tiling_vars.extend((
                        "_tilelang_fallback_x0block",
                        "_tilelang_fallback_x0block_sub",
                    ))
                    code.writeline(
                        "_tilelang_fallback_x0block = ("
                        f"_tilelang_configs[0]['X0BLOCK'] "
                        f"if _tilelang_configs else {fallback_xblock_hint!r})"
                    )
                    code.writeline(
                        "_tilelang_fallback_x0block_sub = ("
                        "_tilelang_configs[0].get('X0BLOCK_SUB', "
                        "_tilelang_fallback_x0block) "
                        f"if _tilelang_configs else {fallback_xblock_hint!r})"
                    )
                    if is_reduction_kernel:
                        fallback_rblock_hint = _DEFAULT_REDUCTION_RBLOCK
                        fallback_tiling_vars.append(
                            "_tilelang_fallback_r0block"
                        )
                        code.writeline(
                            "_tilelang_fallback_r0block = ("
                            "_tilelang_configs[0].get('R0BLOCK', "
                            f"{fallback_rblock_hint!r}) "
                            f"if _tilelang_configs else {fallback_rblock_hint!r})"
                        )
                code.writeline(
                    "_tilelang_fallback_multibuffer = bool("
                    "_tilelang_configs[0].get('multibuffer', False)) "
                    "if _tilelang_configs else False"
                )
                code.writeline(
                    f"if {autotune_enabled_fn}() and len(_tilelang_configs) > 1:"
                )
                with code.indent():
                    code.writeline("try:")
                    with code.indent():
                        code.writeline("import os as _tilelang_os")
                        code.writeline("import tempfile as _tilelang_tempfile")
                        code.writeline(
                            "_tilelang_cache_dir = _tilelang_os.path.abspath("
                            "_tilelang_os.environ.get('TILELANG_CACHE_DIR') "
                            "or _tilelang_os.path.join(_tilelang_tempfile.gettempdir(), "
                            "'tilelang_cache'))"
                        )
                        code.writeline(
                            "_tilelang_os.makedirs(_tilelang_cache_dir, exist_ok=True)"
                        )
                        code.writeline(
                            "_tilelang_os.environ['TILELANG_CACHE_DIR'] = "
                            "_tilelang_cache_dir"
                        )
                        code.writeline(
                            "_tilelang_os.environ['TILELANG_BENCH_METHOD'] = 'npu'"
                        )
                        code.writeline("from tilelang.autotuner import AutoTuner as _TileLangAutoTuner")
                        autotune_factory_params = ", ".join([
                            *tiling_arg_names,
                            "multibuffer",
                        ])
                        code.writeline(f"def _autotune_factory({autotune_factory_params}):")
                        with code.indent():
                            autotune_factory_args = ", ".join(
                                [
                                    arg for arg in [
                                        shape_factory_call,
                                        *tiling_arg_names,
                                    ] if arg
                                ]
                            )
                            code.writeline(f"return {factory_fn}({autotune_factory_args})")
                        code.writeline("def _tilelang_supply(_params, config=None):")
                        with code.indent():
                            code.writeline(f"return [{', '.join(profile_input_args)}]")
                        code.writeline(
                            "_tilelang_warmup = "
                            f"{autotune_warmup_fn}()"
                        )
                        code.writeline(
                            "_tilelang_rep = "
                            f"{autotune_rep_fn}()"
                        )
                        code.writeline(
                            "_tilelang_timeout = "
                            f"{autotune_timeout_fn}()"
                        )
                        code.writeline("_tilelang_results = []")
                        code.writeline(
                            "for _tilelang_multibuffer in (False, True):"
                        )
                        with code.indent():
                            code.writeline(
                                "_tilelang_group_configs = ["
                                "_config for _config in _tilelang_configs "
                                "if bool(_config.get('multibuffer', False)) "
                                "== _tilelang_multibuffer]"
                            )
                            code.writeline("if not _tilelang_group_configs:")
                            with code.indent():
                                code.writeline("continue")
                            code.writeline("try:")
                            with code.indent():
                                code.writeline(
                                    "_tilelang_tuner = "
                                    "_TileLangAutoTuner.from_kernel("
                                    "_autotune_factory, "
                                    "_tilelang_group_configs)"
                                )
                                code.writeline(
                                    "_tilelang_tuner.set_compile_args("
                                )
                                with code.indent():
                                    code.writeline(
                                        f"out_idx={output_arg_indices!r}, "
                                        "target='ascend', execution_backend='tvm_ffi', "
                                        f"pass_configs={ascend_pass_configs_fn}("
                                        "_tilelang_multibuffer)"
                                    )
                                code.writeline(")")
                                code.writeline(
                                    "_tilelang_tuner.set_profile_args("
                                )
                                with code.indent():
                                    code.writeline(
                                        "skip_check=True, "
                                        "cache_input_tensors=False, "
                                        "supply_prog=_tilelang_supply, "
                                        "warmup=_tilelang_warmup, "
                                        "rep=_tilelang_rep, "
                                        "timeout=_tilelang_timeout"
                                    )
                                code.writeline(")")
                                code.writeline(
                                    "_tilelang_results.append("
                                    "_tilelang_tuner.run("
                                    "warmup=_tilelang_warmup, "
                                    "rep=_tilelang_rep, "
                                    "timeout=_tilelang_timeout))"
                                )
                            code.writeline(
                                "except Exception as "
                                "_tilelang_group_autotune_exc:"
                            )
                            with code.indent():
                                code.writeline(
                                    "print('TileLang autotune group failed', "
                                    "{'multibuffer': "
                                    "_tilelang_multibuffer}, "
                                    "_tilelang_group_autotune_exc)"
                                )
                        code.writeline("if not _tilelang_results:")
                        with code.indent():
                            code.writeline(
                                "raise RuntimeError("
                                "'no TileLang autotune group succeeded')"
                            )
                        code.writeline(
                            "_tilelang_result = min("
                            "_tilelang_results, "
                            "key=lambda _result: float("
                            "getattr(_result, 'latency', float('inf'))))"
                        )
                        code.writeline(
                            "_tilelang_selected_config = getattr(_tilelang_result, 'config', None) or {}"
                        )
                        code.writeline(
                            "_tilelang_selected_multibuffer = bool("
                            "_tilelang_selected_config.get('multibuffer', "
                            "_tilelang_fallback_multibuffer))"
                        )
                        selected_tiling_vars: list[str] = []
                        for tiling_arg_name, fallback_var in zip(
                            tiling_arg_names,
                            fallback_tiling_vars,
                        ):
                            selected_var = (
                                f"_tilelang_selected_{tiling_arg_name.lower()}"
                            )
                            selected_tiling_vars.append(selected_var)
                            code.writeline(
                                f"{selected_var} = _tilelang_selected_config.get("
                                f"{tiling_arg_name!r}, {fallback_var})"
                            )
                        selected_factory_args = ", ".join(
                            [
                                arg for arg in [
                                    shape_factory_call,
                                    *selected_tiling_vars,
                                ] if arg
                            ]
                        )
                        code.writeline(f"_compiled_kernel = {import_alias}.compile(")
                        with code.indent():
                            code.writeline(
                                f"{factory_fn}({selected_factory_args}), target='ascend', execution_backend='tvm_ffi', "
                                f"pass_configs={ascend_pass_configs_fn}("
                                "_tilelang_selected_multibuffer)"
                            )
                        code.writeline(")")
                        code.writeline(
                            "print('TileLang autotune selected', "
                            "_tilelang_selected_config)"
                        )
                    code.writeline("except Exception as _tilelang_autotune_exc:")
                    with code.indent():
                        code.writeline(
                            "print('TileLang autotune failed, falling back to default compile:', "
                            "_tilelang_autotune_exc)"
                        )
                        code.writeline("_compiled_kernel = None")
                code.writeline("if _compiled_kernel is None:")
                with code.indent():
                    fallback_factory_args = ", ".join(
                        [
                            arg for arg in [
                                shape_factory_call,
                                *fallback_tiling_vars,
                            ] if arg
                        ]
                    )
                    code.writeline(f"_compiled_kernel = {import_alias}.compile(")
                    with code.indent():
                        code.writeline(
                            f"{factory_fn}({fallback_factory_args}), target='ascend', execution_backend='tvm_ffi', "
                            f"pass_configs={ascend_pass_configs_fn}("
                            "_tilelang_fallback_multibuffer)"
                        )
                    code.writeline(")")
                code.writeline(f"{cache_var}[_key] = _compiled_kernel")
            code.writeline(f"{cache_var}[_key]({', '.join(tensor_call_args)})")

        wrapper.header.splice(code.getvalue())
        return kernel_name, src_code
