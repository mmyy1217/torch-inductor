# TileLang plan capture and code generation for the Ascend 950 backend.
# The A5 pointwise path emits tilelang.ascend.language, explicit SIMD VFs,
# and caller-owned outputs compiled through target="ascend".
# Reduction planning is retained for the next migration stage.
from __future__ import annotations

import dataclasses
import functools
import os
import re
from typing import Any, Callable, Optional, Sequence, cast

import sympy
import torch
from torch.utils._sympy.functions import FloorDiv, ModularIndexing

from torch._inductor import config, ir
from torch._inductor.codegen.common import (
    CSE,
    CSEVariable,
    IndentedBuffer,
    OpOverrides,
    TensorArg,
)
from torch._inductor.codegen.simd import (
    SIMDKernel,
    IterationRangesRoot,
    IterationRangesEntry,
)
from torch._inductor.runtime.hints import ReductionHint
from torch._inductor.utils import Placeholder, sympy_product
from torch._inductor.virtualized import ReductionType, StoreMode, V

from torch_npu._inductor.runtime.tile_generator import TileGenerator
from torch_npu._inductor.codegen.triton import (
    IterationRangesRootNPUIndex,
    NPUIndexTritonKernel,
)
from torch_npu._inductor.codegen.triton_utils import NPUKernelType


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

_TORCH_TO_TL_DTYPE: dict[torch.dtype, str] = {
    torch.float16:  "float16",
    torch.bfloat16: "bfloat16",
    torch.float32:  "float32",
    torch.float64:  "float64",
    torch.int8:     "int8",
    torch.int16:    "int16",
    torch.int32:    "int32",
    torch.int64:    "int64",
    torch.uint8:    "uint8",
    torch.bool:     "bool",
}


def tilelang_dtype(dtype: torch.dtype) -> str:
    return _TORCH_TO_TL_DTYPE.get(dtype, "float32")


def _tilelang_alloc_shared(name: str, shape: str, dtype: torch.dtype) -> str:
    return f"{name} = T.alloc_shared({shape}, '{tilelang_dtype(dtype)}')"


def _tilelang_shape_tuple_expr(dims: Sequence[Any]) -> str:
    rendered_dims = [str(dim) for dim in dims]
    suffix = "," if len(rendered_dims) == 1 else ""
    return f"({', '.join(rendered_dims)}{suffix})"


def _tilelang_slice(start: Any, extent: Any) -> str:
    start_text = str(start)
    extent_text = str(extent)
    return f"{start_text}:{start_text} + {extent_text}"


def _tilelang_local_slice(extent: Any) -> str:
    return f"0:{extent}"


def _tilelang_ref(name: str, indices: Sequence[Any]) -> str:
    if not indices:
        return name
    return f"{name}[{', '.join(str(index) for index in indices)}]"


def _tilelang_copy(src: str, dst: str) -> str:
    return f"T.copy({src}, {dst})"


def _split_tilelang_shape_expr(shape: str) -> tuple[str, ...]:
    shape = shape.strip()
    if not (shape.startswith("(") and shape.endswith(")")):
        return (shape,)
    inner = shape[1:-1].strip()
    if not inner:
        return ()

    parts: list[str] = []
    depth = 0
    start = 0
    for i, char in enumerate(inner):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            part = inner[start:i].strip()
            if part:
                parts.append(part)
            start = i + 1
    tail = inner[start:].strip()
    if tail:
        parts.append(tail)
    return tuple(parts)


def _split_tilelang_index_tuple(indices: str) -> tuple[str, ...]:
    return _split_tilelang_shape_expr(f"({indices})")


def _tilelang_dim_label(dim: Any, fallback_index: int) -> str:
    dim_text = str(dim)
    if "_rnumel" in dim_text or dim_text == "_RBLOCK":
        return "r"
    match = re.search(r"_([wvutzyx])(?:numel|BLOCK|BLOCK_SUB)?", dim_text)
    if match:
        return match.group(1)
    if dim_text in {"1", "0"}:
        return f"d{fallback_index}"
    return dim_text.lstrip("_") or f"d{fallback_index}"


@dataclasses.dataclass(frozen=True)
class TileLangTensor:
    """Shape/dtype view of one TileLang tensor value."""

    dims: tuple[str, ...]
    current_shape: tuple[str, ...]
    dtype: torch.dtype

    def __post_init__(self) -> None:
        if len(self.dims) != len(self.current_shape):
            raise ValueError(
                "TileLangTensor dims and current_shape must have the same rank: "
                f"{self.dims} vs {self.current_shape}"
            )

    @classmethod
    def from_shape(
        cls,
        shape: "TileLangTensor | Sequence[Any] | str",
        dtype: torch.dtype,
        dims: Optional[Sequence[Any]] = None,
    ) -> "TileLangTensor":
        if isinstance(shape, TileLangTensor):
            return dataclasses.replace(shape, dtype=dtype)

        current_shape = (
            _split_tilelang_shape_expr(shape)
            if isinstance(shape, str)
            else tuple(str(dim) for dim in shape)
        )
        if dims is None:
            dims = [
                _tilelang_dim_label(dim, i)
                for i, dim in enumerate(current_shape)
            ]
        return cls(
            dims=tuple(str(dim) for dim in dims),
            current_shape=current_shape,
            dtype=dtype,
        )

    @property
    def shape(self) -> str:
        return _tilelang_shape_tuple_expr(self.current_shape)

    @property
    def dtype_str(self) -> str:
        return tilelang_dtype(self.dtype)

    def tensor_type(self) -> str:
        return f"T.Tensor({self.shape}, '{self.dtype_str}')"

    def ref(
        self,
        name: str,
        indices: Sequence[Any] | dict[str, Any],
    ) -> str:
        if isinstance(indices, dict):
            index_map = {str(dim): str(index) for dim, index in indices.items()}
            indices = [index_map[dim] for dim in self.dims]
        return _tilelang_ref(name, indices)

    def tiled_ref(
        self,
        name: str,
        loops: dict[str, "TileLangForLoop"],
        *,
        local: bool,
        dim_extents: Optional[dict[str, Any]] = None,
        singleton_as_slice: bool = True,
        singleton_index_dims: Optional[set[str]] = None,
    ) -> str:
        dim_extents = {
            str(dim): str(extent)
            for dim, extent in (dim_extents or {}).items()
        }
        singleton_index_dims = singleton_index_dims or set()
        loop_map = {str(dim): loop for dim, loop in loops.items()}
        indices: list[str] = []
        for dim, extent in zip(self.dims, self.current_shape):
            tile_extent = dim_extents.get(dim)
            if TileLangBlock._is_broadcast_extent(extent):
                indices.append(
                    "0"
                    if dim in singleton_index_dims or not singleton_as_slice
                    else "0:1"
                )
                continue
            loop = loop_map.get(dim)
            if loop is not None:
                indices.append(
                    loop.local_slice(tile_extent)
                    if local else loop.global_slice(tile_extent)
                )
                continue
            indices.append(_tilelang_local_slice(tile_extent or extent))
        return self.ref(name, indices)

    def _materialized_shape(
        self,
        block: Optional["TileLangBlock"] = None,
        dim_extents: Optional[dict[str, Any]] = None,
    ) -> tuple[str, ...]:
        dim_extents = {
            str(dim): str(extent)
            for dim, extent in (dim_extents or {}).items()
        }
        return tuple(
            dim_extents[dim]
            if dim in dim_extents
            else (
                block.extent_for(dim, extent)
                if block is not None
                else extent
            )
            for dim, extent in zip(self.dims, self.current_shape)
        )

    def shape_expr(
        self,
        block: Optional["TileLangBlock"] = None,
        dim_extents: Optional[dict[str, Any]] = None,
    ) -> str:
        return _tilelang_shape_tuple_expr(
            self._materialized_shape(block, dim_extents)
        )

    def shape_list_expr(
        self,
        block: Optional["TileLangBlock"] = None,
        dim_extents: Optional[dict[str, Any]] = None,
    ) -> str:
        return f"[{', '.join(self._materialized_shape(block, dim_extents))}]"

    def with_dim_extents(self, dim_extents: dict[str, Any]) -> "TileLangTensor":
        dim_extents = {
            str(dim): str(extent)
            for dim, extent in dim_extents.items()
        }
        return dataclasses.replace(
            self,
            current_shape=tuple(
                dim_extents.get(dim, extent)
                for dim, extent in zip(self.dims, self.current_shape)
            ),
        )


@dataclasses.dataclass(frozen=True)
class TileLangBlock:
    """Kernel block shape for named logical tensor dimensions."""

    name: str
    dim_blocks: tuple[tuple[str, str], ...]

    @classmethod
    def create(cls, name: str, dim_blocks: dict[str, Any]) -> "TileLangBlock":
        return cls(
            name=name,
            dim_blocks=tuple((str(dim), str(block)) for dim, block in dim_blocks.items()),
        )

    def _block_map(self) -> dict[str, str]:
        return dict(self.dim_blocks)

    @staticmethod
    def _is_broadcast_extent(extent: str) -> bool:
        return extent.strip() in {"1", "T.IntImm('int32', 1)"}

    def extent_for(self, dim: str, extent: str) -> str:
        block_map = self._block_map()
        if dim in block_map and not self._is_broadcast_extent(extent):
            return block_map[dim]
        return extent


@dataclasses.dataclass(frozen=True)
class TileLangForLoop:
    """Named loop semantics for one tiled dimension."""

    name: str
    extent: str
    block: str
    loop_var: str
    base: str
    remain: str
    base_expr: Optional[str] = None

    def emit_base(self, code: IndentedBuffer, prefix: str = "0") -> None:
        code.writeline(f"{self.base} = {prefix}")

    def emit_full_remain(self, code: IndentedBuffer) -> None:
        code.writeline(f"{self.remain} = {self.extent}")

    def emit_first_remain(self, code: IndentedBuffer) -> None:
        code.writeline(f"{self.remain} = T.min({self.block}, {self.extent})")

    def emit_serial_header(self, code: IndentedBuffer, start: int = 0) -> None:
        start_part = "" if start == 0 else f"{start}, "
        code.writeline(
            f"for {self.loop_var} in T.serial({start_part}T.ceildiv({self.extent}, {self.block})):"
        )

    def emit_serial_base(self, code: IndentedBuffer) -> None:
        base_expr = self.base_expr or f"{self.loop_var} * {self.block}"
        code.writeline(f"{self.base} = {base_expr}")

    def emit_serial_remain(self, code: IndentedBuffer) -> None:
        code.writeline(f"{self.remain} = T.min({self.block}, {self.extent} - {self.base})")

    def local_slice(self, extent: Optional[Any] = None) -> str:
        return _tilelang_local_slice(extent or self.remain)

    def global_slice(self, extent: Optional[Any] = None) -> str:
        return _tilelang_slice(self.base, extent or self.remain)


# ---------------------------------------------------------------------------
# NPU vector operation model
#
# Dtype support is sourced from tilelang-mlir-ascend/docs/Tilelang.language/.
# Only dtypes reachable through inductor are listed here.
# ---------------------------------------------------------------------------

_FP = frozenset({torch.float16, torch.bfloat16, torch.float32})
_FP_INT = frozenset({
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int16,
    torch.int32,
    torch.int64,
})
_CMP = frozenset({
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.float32,
    torch.bfloat16,
})
_PREDICATE_DISABLED_REASON = (
    "predicate intrinsics are not runtime-stable on the current Ascend stack"
)
_UNVALIDATED_VECTOR_OP_REASON = "the intrinsic has not passed runtime validation"


@dataclasses.dataclass(frozen=True)
class TileLangOpSpec:
    """Complete lowering contract for one primitive vector operation."""

    name: str
    intrinsic: str
    arity: int
    supported_dtypes: Optional[frozenset[torch.dtype]]
    result_kind: str = "same"
    predicate_operands: tuple[int, ...] = ()
    comparison_mode: Optional[str] = None
    scalar_rewrite: Optional[str] = None
    enabled: bool = True
    disabled_reason: str = ""

    def validate_dtype(self, dtype: torch.dtype) -> None:
        if not self.enabled:
            raise NotImplementedError(
                f"TileLang NPU: op '{self.name}' (T.{self.intrinsic}) is "
                f"disabled: {self.disabled_reason}. Falling back to Triton."
            )
        if self.supported_dtypes is None or dtype in self.supported_dtypes:
            return
        raise NotImplementedError(
            f"TileLang NPU: op '{self.name}' (T.{self.intrinsic}) "
            f"does not support dtype {dtype}; supported: "
            f"{self.supported_dtypes}. Falling back to Triton."
        )

    def result_dtype(
        self,
        execution_dtype: torch.dtype,
        recorded_dtype: Optional[torch.dtype],
        output_dtype: Optional[torch.dtype],
    ) -> torch.dtype:
        if self.result_kind == "bool":
            return torch.bool
        return recorded_dtype or output_dtype or execution_dtype

    def lower(
        self,
        operands: Sequence[Any],
        execution_dtype: torch.dtype,
    ) -> tuple["TileLangOpSpec", tuple[Any, ...]]:
        lowered_operands = tuple(operands)
        if self.scalar_rewrite == "neg":
            return _tilelang_op("mul"), (*lowered_operands, -1.0)
        if (
            self.scalar_rewrite == "reciprocal"
            and execution_dtype in _FP
            and len(lowered_operands) == 2
            and _is_scalar_literal_operand(lowered_operands[1])
            and float(lowered_operands[1]) != 0.0
        ):
            return _tilelang_op("mul"), (
                lowered_operands[0],
                1.0 / float(lowered_operands[1]),
            )
        return self, lowered_operands

    def emit_call(self, operands: Sequence[Any], out_buf: str) -> str:
        if len(operands) != self.arity:
            raise RuntimeError(
                f"TileLang op '{self.name}' expects {self.arity} operands, "
                f"got {len(operands)}"
            )
        resolved = [_resolve_operand(operand) for operand in operands]
        if self.intrinsic == "copy":
            return _tilelang_copy(resolved[0], out_buf)
        arguments = [*resolved, out_buf]
        if self.comparison_mode is not None:
            arguments.append(repr(self.comparison_mode))
        return f"T.{self.intrinsic}({', '.join(arguments)})"


def _op(
    name: str,
    intrinsic: str,
    arity: int,
    supported_dtypes: Optional[frozenset[torch.dtype]],
    **kwargs: Any,
) -> TileLangOpSpec:
    return TileLangOpSpec(
        name=name,
        intrinsic=intrinsic,
        arity=arity,
        supported_dtypes=supported_dtypes,
        **kwargs,
    )


_TILELANG_OPS: dict[str, TileLangOpSpec] = {
    "copy": _op("copy", "copy", 1, None),
    "add": _op("add", "vadd", 2, _FP),
    "sub": _op("sub", "vsub", 2, _FP),
    "mul": _op("mul", "vmul", 2, _FP_INT),
    "truediv": _op(
        "truediv",
        "vdiv",
        2,
        frozenset({torch.float16, torch.float32, torch.int64}),
        scalar_rewrite="reciprocal",
    ),
    "maximum": _op("maximum", "vmax", 2, _FP),
    "minimum": _op(
        "minimum",
        "vmin",
        2,
        frozenset({
            torch.float16,
            torch.float32,
            torch.bfloat16,
            torch.int16,
            torch.int32,
            torch.int64,
        }),
    ),
    "bitwise_and": _op(
        "bitwise_and",
        "vand",
        2,
        frozenset({
            torch.int8,
            torch.int64,
            torch.float16,
            torch.float32,
            torch.bool,
        }),
    ),
    "bitwise_or": _op("bitwise_or", "vor", 2, frozenset()),
    "bitwise_xor": _op("bitwise_xor", "vxor", 2, frozenset()),
    "bitwise_not": _op(
        "bitwise_not",
        "vnot",
        1,
        frozenset({
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.bool,
        }),
        enabled=False,
        disabled_reason=_UNVALIDATED_VECTOR_OP_REASON,
    ),
    "logical_not": _op(
        "logical_not",
        "vnot",
        1,
        frozenset({torch.bool}),
        result_kind="bool",
        enabled=False,
        disabled_reason=_PREDICATE_DISABLED_REASON,
    ),
    "logical_and": _op(
        "logical_and",
        "vand",
        2,
        frozenset({torch.bool}),
        result_kind="bool",
        enabled=False,
        disabled_reason=_PREDICATE_DISABLED_REASON,
    ),
    "logical_or": _op(
        "logical_or",
        "vor",
        2,
        frozenset({torch.bool}),
        result_kind="bool",
        enabled=False,
        disabled_reason=_PREDICATE_DISABLED_REASON,
    ),
    "logical_xor": _op(
        "logical_xor",
        "vxor",
        2,
        frozenset({torch.bool}),
        result_kind="bool",
        enabled=False,
        disabled_reason=_PREDICATE_DISABLED_REASON,
    ),
    "where": _op(
        "where",
        "vselect",
        3,
        _FP,
        predicate_operands=(0,),
        enabled=False,
        disabled_reason=_PREDICATE_DISABLED_REASON,
    ),
    "neg": _op("neg", "vmul", 1, _FP_INT, scalar_rewrite="neg"),
    "exp": _op("exp", "vexp", 1, _FP),
    "log": _op("log", "vln", 1, _FP),
    "relu": _op("relu", "vrelu", 1, _FP),
    "sigmoid": _op("sigmoid", "vsigmoid", 1, _FP),
    "sqrt": _op("sqrt", "vsqrt", 1, _FP),
    "rsqrt": _op("rsqrt", "vrsqrt", 1, _FP),
    "abs": _op(
        "abs",
        "vabs",
        1,
        frozenset({
            torch.float16,
            torch.float32,
            torch.uint8,
            torch.int32,
            torch.int64,
        }),
    ),
    "cos": _op("cos", "vcos", 1, _FP),
    "sin": _op("sin", "vsin", 1, _FP),
    "erf": _op("erf", "verf", 1, _FP),
    "tanh": _op("tanh", "vtanh", 1, _FP),
}

for _comparison in ("lt", "le", "gt", "ge", "eq", "ne"):
    _TILELANG_OPS[_comparison] = _op(
        _comparison,
        "vcmp",
        2,
        _CMP,
        result_kind="bool",
        comparison_mode=_comparison,
        enabled=False,
        disabled_reason=_PREDICATE_DISABLED_REASON,
    )


def _tilelang_op(op_name: str) -> TileLangOpSpec:
    try:
        return _TILELANG_OPS[op_name]
    except KeyError as exc:
        raise NotImplementedError(
            f"TileLang NPU backend: op '{op_name}' has no T.v* equivalent. "
            "This kernel will fall back to Triton."
        ) from exc


_ANY_SUPPORTED_DTYPE: frozenset[torch.dtype] = frozenset().union(
    *(
        spec.supported_dtypes
        for spec in _TILELANG_OPS.values()
        if spec.supported_dtypes is not None
    )
)


# ---------------------------------------------------------------------------
# CSE variable
# ---------------------------------------------------------------------------

class TileLangCSEVariable(CSEVariable):
    pass


@dataclasses.dataclass(frozen=True)
class PendingOp:
    """Explicit metadata for the next CSE variable produced by an override."""

    name: str
    operands: tuple[Any, ...]
    dtype: Optional[torch.dtype] = None

    @classmethod
    def create(
        cls,
        name: str,
        operands: Sequence[Any],
        dtype: Optional[torch.dtype] = None,
    ) -> "PendingOp":
        return cls(name=name, operands=tuple(operands), dtype=dtype)


@dataclasses.dataclass(frozen=True)
class VecOp:
    """A typed TileLang vector operation in topological emit order."""

    spec: TileLangOpSpec
    operands: tuple[Any, ...]
    out: str
    dtype: Optional[torch.dtype] = None
    execution_dtype: Optional[torch.dtype] = None

    @classmethod
    def create(
        cls,
        name: str,
        operands: Sequence[Any],
        out: str,
        dtype: Optional[torch.dtype] = None,
    ) -> "VecOp":
        spec = _tilelang_op(name)
        if len(operands) != spec.arity:
            raise RuntimeError(
                f"TileLang op '{name}' expects {spec.arity} operands, "
                f"got {len(operands)}"
            )
        return cls(spec, tuple(operands), out, dtype)

    @property
    def name(self) -> str:
        return self.spec.name

    def with_out(self, out: str) -> "VecOp":
        return dataclasses.replace(self, out=out)

    def with_dtypes(
        self,
        execution_dtype: torch.dtype,
        result_dtype: torch.dtype,
    ) -> "VecOp":
        return dataclasses.replace(
            self,
            dtype=result_dtype,
            execution_dtype=execution_dtype,
        )

    def lower(
        self,
        operands: Sequence[Any],
        out: str,
    ) -> "VecOp":
        if self.execution_dtype is None or self.dtype is None:
            raise RuntimeError("VecOp dtypes must be normalized before planning")
        spec, lowered_operands = self.spec.lower(
            operands,
            self.execution_dtype,
        )
        return dataclasses.replace(
            self,
            spec=spec,
            operands=lowered_operands,
            out=out,
        )


@dataclasses.dataclass(frozen=True)
class OperationOrigin:
    """Capture metadata retained for planning and fallback diagnostics."""

    kind: str
    name: str
    detail: str = ""

    def __str__(self) -> str:
        origin = f"{self.kind}:{self.name}"
        if self.detail:
            origin = f"{origin} ({self.detail})"
        return origin


class TileLangUnsupported(NotImplementedError):
    """Structured, pre-emission rejection of an unsupported TileLang plan."""

    def __init__(
        self,
        stage: str,
        reason: str,
        operation: Optional[str] = None,
        origin: Optional[OperationOrigin] = None,
    ) -> None:
        self.stage = stage
        self.reason = reason
        self.operation = operation
        self.origin = origin
        details = [f"TileLang {stage}: {reason}"]
        if operation is not None:
            details.append(f"operation={operation}")
        if origin is not None:
            details.append(f"origin={origin}")
        super().__init__("; ".join(details))


@dataclasses.dataclass(frozen=True)
class BufferPlan:
    """One physical shared-buffer allocation."""

    name: str
    shape: str
    dtype: torch.dtype
    scope: str = "shared"


@dataclasses.dataclass(frozen=True)
class TransferPlan:
    kind: str
    source: str
    destination: str
    extent: Optional[str] = None
    source_index: Optional[str] = None
    induction_var: str = "_tl_i"


@dataclasses.dataclass(frozen=True)
class VecRegionPlan:
    operations: tuple[VecOp, ...]


@dataclasses.dataclass(frozen=True)
class OperationBase:
    op_id: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    origin: OperationOrigin


@dataclasses.dataclass(frozen=True)
class PointwiseOpPlan(OperationBase):
    compute: VecRegionPlan


@dataclasses.dataclass(frozen=True)
class ReductionOpPlan(OperationBase):
    source: str
    source_tensor: TileLangTensor
    accumulator: str
    reduction_type: str
    reduce_dims: tuple[str, ...]
    size: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class BroadcastOpPlan(OperationBase):
    source: str
    destination: str
    source_tensor: TileLangTensor
    destination_tensor: TileLangTensor
    broadcast_dims: tuple[str, ...]
    intrinsic: str = "vbrc"


KernelOperationPlan = PointwiseOpPlan | ReductionOpPlan | BroadcastOpPlan


@dataclasses.dataclass(frozen=True)
class BroadcastRequest:
    source: str
    destination: str
    source_tensor: TileLangTensor
    destination_tensor: TileLangTensor
    broadcast_dims: tuple[str, ...]
    origin: OperationOrigin
    consumer_id: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class TileLangOperationGraph:
    operations: tuple[KernelOperationPlan, ...]
    external_values: frozenset[str]

    def validate(self) -> None:
        operation_ids: set[str] = set()
        producers: dict[str, str] = {}
        positions: dict[str, int] = {}
        for position, operation in enumerate(self.operations):
            if operation.op_id in operation_ids:
                raise TileLangUnsupported(
                    "operation-graph",
                    "operation ids must be unique",
                    operation.op_id,
                    operation.origin,
                )
            operation_ids.add(operation.op_id)
            positions[operation.op_id] = position
            for output in operation.outputs:
                if output in producers or output in self.external_values:
                    raise TileLangUnsupported(
                        "operation-graph",
                        f"logical buffer '{output}' has more than one producer",
                        operation.op_id,
                        operation.origin,
                    )
                producers[output] = operation.op_id

        dependencies: dict[str, set[str]] = {
            operation.op_id: set() for operation in self.operations
        }
        for operation in self.operations:
            for input_name in operation.inputs:
                producer = producers.get(input_name)
                if producer is None:
                    if input_name not in self.external_values:
                        raise TileLangUnsupported(
                            "operation-graph",
                            f"input '{input_name}' has no producer",
                            operation.op_id,
                            operation.origin,
                        )
                    continue
                dependencies[operation.op_id].add(producer)

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(op_id: str) -> None:
            if op_id in visiting:
                raise TileLangUnsupported(
                    "operation-graph",
                    "operation graph contains a cycle",
                    op_id,
                )
            if op_id in visited:
                return
            visiting.add(op_id)
            for dependency in dependencies[op_id]:
                visit(dependency)
            visiting.remove(op_id)
            visited.add(op_id)

        for operation in self.operations:
            visit(operation.op_id)
        for operation in self.operations:
            for producer in dependencies[operation.op_id]:
                if positions[producer] >= positions[operation.op_id]:
                    raise TileLangUnsupported(
                        "operation-graph",
                        f"producer '{producer}' must precede its consumer",
                        operation.op_id,
                        operation.origin,
                    )


class BroadcastInsertionPass:
    """Materializes explicit BroadcastOp nodes before their consumers."""

    @staticmethod
    def _validate(request: BroadcastRequest) -> None:
        source = request.source_tensor
        destination = request.destination_tensor
        if source.dtype != destination.dtype:
            raise TileLangUnsupported(
                "broadcast-insertion",
                "broadcast source and destination dtypes must match",
                origin=request.origin,
            )
        if source.dims != destination.dims:
            raise TileLangUnsupported(
                "broadcast-insertion",
                "broadcast source and destination dimensions must align",
                origin=request.origin,
            )

        broadcast_dims = set(request.broadcast_dims)
        expanded_dims: set[str] = set()
        for dim, source_extent, destination_extent in zip(
            source.dims,
            source.current_shape,
            destination.current_shape,
        ):
            if source_extent == destination_extent:
                continue
            if not TileLangBlock._is_broadcast_extent(source_extent):
                raise TileLangUnsupported(
                    "broadcast-insertion",
                    f"dimension '{dim}' is not a singleton source extent",
                    origin=request.origin,
                )
            expanded_dims.add(dim)
        if expanded_dims != broadcast_dims:
            raise TileLangUnsupported(
                "broadcast-insertion",
                "broadcast_dims must exactly describe singleton expansion",
                origin=request.origin,
            )

    def run(
        self,
        operations: Sequence[KernelOperationPlan],
        external_values: set[str],
        requests: Sequence[BroadcastRequest],
    ) -> TileLangOperationGraph:
        operation_ids = {operation.op_id for operation in operations}
        broadcasts_by_consumer: dict[Optional[str], list[BroadcastOpPlan]] = {}
        seen: set[tuple[Any, ...]] = set()
        for request in requests:
            self._validate(request)
            if (
                request.consumer_id is not None
                and request.consumer_id not in operation_ids
            ):
                raise TileLangUnsupported(
                    "broadcast-insertion",
                    f"unknown broadcast consumer '{request.consumer_id}'",
                    origin=request.origin,
                )
            key = (
                request.source,
                request.destination,
                request.source_tensor,
                request.destination_tensor,
                request.destination_tensor.dtype,
                request.consumer_id,
            )
            if key in seen:
                continue
            seen.add(key)
            consumer_broadcasts = broadcasts_by_consumer.setdefault(
                request.consumer_id,
                [],
            )
            broadcast_index = sum(
                len(items) for items in broadcasts_by_consumer.values()
            )
            consumer_broadcasts.append(
                BroadcastOpPlan(
                    op_id=f"broadcast_{broadcast_index}",
                    inputs=(request.source,),
                    outputs=(request.destination,),
                    origin=request.origin,
                    source=request.source,
                    destination=request.destination,
                    source_tensor=request.source_tensor,
                    destination_tensor=request.destination_tensor,
                    broadcast_dims=request.broadcast_dims,
                )
            )

        scheduled_operations: list[KernelOperationPlan] = []
        for operation in operations:
            scheduled_operations.extend(
                broadcasts_by_consumer.get(operation.op_id, ())
            )
            scheduled_operations.append(operation)
        scheduled_operations.extend(broadcasts_by_consumer.get(None, ()))
        graph = TileLangOperationGraph(
            operations=tuple(scheduled_operations),
            external_values=frozenset(external_values),
        )
        graph.validate()
        return graph


@dataclasses.dataclass(frozen=True)
class KernelRegionPlan:
    region_id: str
    loops: tuple[TileLangForLoop, ...]
    block: TileLangBlock
    allocations: tuple[BufferPlan, ...]
    input_transfers: tuple[TransferPlan, ...]
    operations: tuple[str, ...]
    output_transfers: tuple[TransferPlan, ...]
    prologue: tuple[str, ...] = ()
    body_prologue: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class XTileRegionPlan(KernelRegionPlan):
    serial_loop: Optional[TileLangForLoop] = None
    subregions: tuple[KernelRegionPlan, ...] = ()


@dataclasses.dataclass(frozen=True)
class RTileIterationPlan:
    loop: TileLangForLoop
    mode: str
    reduction_ops: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class RTileRegionPlan(KernelRegionPlan):
    iteration: Optional[RTileIterationPlan] = None


@dataclasses.dataclass(frozen=True)
class KernelSignaturePlan:
    prim_fn_name: str
    parameters: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class KernelLaunchPlan:
    grid: str
    bindings: str = "cid"


@dataclasses.dataclass(frozen=True)
class TileLangKernelPlan:
    signature: KernelSignaturePlan
    launch: KernelLaunchPlan
    operations: tuple[KernelOperationPlan, ...]
    regions: tuple[KernelRegionPlan, ...]
    preamble: tuple[str, ...] = ()

    def operation(self, op_id: str) -> KernelOperationPlan:
        for operation in self.operations:
            if operation.op_id == op_id:
                return operation
        raise KeyError(f"unknown TileLang operation id: {op_id}")

    def validate(self) -> None:
        def walk_regions(
            regions: Sequence[KernelRegionPlan],
        ) -> list[KernelRegionPlan]:
            flattened: list[KernelRegionPlan] = []
            for region in regions:
                flattened.append(region)
                if isinstance(region, XTileRegionPlan):
                    flattened.extend(walk_regions(region.subregions))
            return flattened

        all_regions = walk_regions(self.regions)
        operation_ids = [operation.op_id for operation in self.operations]
        if len(operation_ids) != len(set(operation_ids)):
            raise RuntimeError("TileLang kernel plan contains duplicate operation ids")
        scheduled_ids = [
            op_id
            for region in all_regions
            for op_id in region.operations
        ]
        if len(scheduled_ids) != len(set(scheduled_ids)):
            raise RuntimeError("TileLang operation is scheduled in multiple regions")
        if set(scheduled_ids) != set(operation_ids):
            raise RuntimeError(
                "TileLang regions must schedule every operation exactly once"
            )
        for region in all_regions:
            if not isinstance(region, RTileRegionPlan):
                continue
            if region.iteration is None:
                raise RuntimeError("RTileRegionPlan requires an iteration plan")
            reduction_ops = tuple(
                op_id
                for op_id in region.operations
                if isinstance(self.operation(op_id), ReductionOpPlan)
            )
            if reduction_ops != region.iteration.reduction_ops:
                raise RuntimeError(
                    "RTile iteration reduction_ops do not match region operations"
                )

        scheduled_position = {
            op_id: position for position, op_id in enumerate(scheduled_ids)
        }
        producers = {
            output: operation.op_id
            for operation in self.operations
            for output in operation.outputs
        }
        for operation in self.operations:
            for input_name in operation.inputs:
                producer = producers.get(input_name)
                if producer is None:
                    continue
                if scheduled_position[producer] >= scheduled_position[operation.op_id]:
                    raise RuntimeError(
                        "TileLang region order violates operation dependencies"
                    )


@dataclasses.dataclass(frozen=True)
class PointwiseLayoutPlan:
    """Shape/index metadata for one pointwise lowering route."""

    kind: str
    full_shape: str
    local_shape: str
    tensor_shapes: dict[str, TileLangTensor]
    dims: tuple[str, ...]
    block: TileLangBlock
    axis_lengths: tuple[sympy.Expr, ...] = ()
    axis_dim_vars: tuple[str, ...] = ()
    axis_block_vars: tuple[str, ...] = ()
    axis_deps: dict[str, tuple[int, ...]] = dataclasses.field(default_factory=dict)
    input_positions: dict[str, list[Optional[int]]] = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass
class TileLangValueInfo:
    """Unified metadata for one CSE value."""

    dtype: Optional[torch.dtype] = None
    op: Optional[PendingOp] = None
    buffer: Optional[str] = None
    const: Optional[Any] = None


@dataclasses.dataclass
class TileLangBufferInfo:
    """Unified metadata for one TileLang input/output local buffer."""

    name: str
    var: str
    local: str
    dtype: torch.dtype
    role: str
    index: Optional[sympy.Expr] = None
    numel: Optional[sympy.Expr] = None


@dataclasses.dataclass
class TileLangOutputBinding:
    """Binding from an output local buffer to the CSE value that fills it."""

    local: str
    value: TileLangCSEVariable
    dtype: torch.dtype
    index: Optional[sympy.Expr] = None


class KernelBufferPlanner:
    """Resolves logical values, physical allocations, and temporary lifetimes."""

    def plan_vec_region(
        self,
        operations: Sequence[VecOp],
        *,
        local_shape: str,
        default_dtype: torch.dtype,
        protected_buffers: set[str],
        buffer_shapes: dict[str, str],
        buffer_dtypes: dict[str, torch.dtype],
    ) -> tuple[VecRegionPlan, tuple[BufferPlan, ...]]:
        use_counts: dict[str, int] = {}
        for operation in operations:
            for operand in operation.operands:
                if isinstance(operand, str):
                    use_counts[operand] = use_counts.get(operand, 0) + 1

        protected = set(protected_buffers)
        allocated = set(protected_buffers)
        shapes = dict(buffer_shapes)
        dtypes = dict(buffer_dtypes)
        assignments: dict[str, str] = {}
        free_buffers: dict[tuple[str, torch.dtype], list[str]] = {}
        allocations: list[BufferPlan] = []
        planned_operations: list[VecOp] = []

        def resolve(operand: Any) -> Any:
            if isinstance(operand, str):
                return assignments.get(operand, operand)
            return operand

        def result_shape(logical_out: str, operands: Sequence[Any]) -> str:
            if logical_out in shapes:
                return shapes[logical_out]
            operand_shapes = [
                shapes[operand]
                for operand in operands
                if isinstance(operand, str) and operand in shapes
            ]
            if local_shape in operand_shapes:
                return local_shape
            if operand_shapes:
                return operand_shapes[0]
            return local_shape

        def assign(logical: str, shape: str, dtype: torch.dtype) -> str:
            if logical in assignments:
                return assignments[logical]
            if logical in allocated:
                physical = logical
            elif free_buffers.get((shape, dtype)):
                physical = free_buffers[(shape, dtype)].pop()
            else:
                physical = logical
                allocations.append(
                    BufferPlan(
                        name=physical,
                        shape=shape,
                        dtype=dtype,
                    )
                )
                allocated.add(physical)
            assignments[logical] = physical
            shapes[logical] = shape
            shapes[physical] = shape
            dtypes[logical] = dtype
            dtypes[physical] = dtype
            return physical

        def release(operand: Any, current_out: str) -> None:
            if not isinstance(operand, str):
                return
            use_counts[operand] = use_counts.get(operand, 0) - 1
            if use_counts[operand] != 0:
                return
            physical = resolve(operand)
            key = (
                shapes.get(physical, local_shape),
                dtypes.get(physical, default_dtype),
            )
            reusable = operand.startswith("_") and operand not in protected
            free_list = free_buffers.setdefault(key, [])
            if (
                reusable
                and physical not in protected
                and physical != current_out
                and physical not in free_list
            ):
                free_list.append(physical)

        for operation in operations:
            if operation.dtype is None or operation.execution_dtype is None:
                raise RuntimeError("VecOp dtypes must be normalized before planning")
            resolved_operands = tuple(resolve(item) for item in operation.operands)
            shape = result_shape(operation.out, resolved_operands)
            out = assign(operation.out, shape, operation.dtype)
            planned = operation.lower(resolved_operands, out)
            planned.spec.validate_dtype(planned.execution_dtype)
            planned_operations.append(planned)
            for operand in operation.operands:
                release(operand, out)
        return (
            VecRegionPlan(tuple(planned_operations)),
            tuple(allocations),
        )


class KernelRegionScheduler:
    """Assigns a validated operation graph to explicit loop regions."""

    @staticmethod
    def schedule_pointwise(
        graph: TileLangOperationGraph,
        region: XTileRegionPlan,
    ) -> tuple[KernelRegionPlan, ...]:
        graph.validate()
        return (
            dataclasses.replace(
                region,
                operations=tuple(operation.op_id for operation in graph.operations),
            ),
        )


@dataclasses.dataclass
class SignatureEmitter:
    code: IndentedBuffer

    def emit(self, signature: KernelSignaturePlan) -> None:
        self.code.writeline("@T.prim_func")
        self.code.writeline(f"def {signature.prim_fn_name}(")
        with self.code.indent():
            for index, parameter in enumerate(signature.parameters):
                comma = "," if index < len(signature.parameters) - 1 else ""
                self.code.writeline(f"{parameter}{comma}")
        self.code.writeline("):")


@dataclasses.dataclass
class TransferEmitter:
    code: IndentedBuffer

    def emit(self, transfer: TransferPlan) -> None:
        if transfer.kind == "COPY":
            self.code.writeline(_tilelang_copy(
                transfer.source,
                transfer.destination,
            ))
            return
        if transfer.kind == "INDEXED_LOAD":
            raise TileLangUnsupported(
                "A5 transfer", "indexed loads require an explicit SIMT route; not enabled"
            )
        raise RuntimeError(f"unknown TileLang transfer kind: {transfer.kind}")


@dataclasses.dataclass
class PointwiseOpEmitter:
    code: IndentedBuffer
    scalar_cache: dict[tuple[str, str], str]

    def _materialize_scalar(
        self,
        operand: Any,
        dtype: torch.dtype,
    ) -> Any:
        if not _is_scalar_literal_operand(operand):
            return operand
        tl_dtype = tilelang_dtype(dtype)
        literal = repr(operand)
        key = (tl_dtype, literal)
        scalar_name = self.scalar_cache.get(key)
        if scalar_name is None:
            scalar_name = f"_tl_scalar_{len(self.scalar_cache)}"
            self.scalar_cache[key] = scalar_name
            self.code.writeline(f"{scalar_name} = T.alloc_var('{tl_dtype}')")
            self.code.writeline(
                f"T.fill({scalar_name}, T.cast({literal}, '{tl_dtype}'))"
            )
        return scalar_name

    def emit(self, operation: PointwiseOpPlan) -> None:
        for vec_operation in operation.compute.operations:
            if vec_operation.execution_dtype is None:
                raise RuntimeError("VecOp execution dtype was not planned")
            operands = [
                self._materialize_scalar(
                    operand,
                    vec_operation.execution_dtype,
                )
                for operand in vec_operation.operands
            ]
            self.code.writeline(
                vec_operation.spec.emit_call(operands, vec_operation.out)
            )


@dataclasses.dataclass
class ReductionOpEmitter:
    code: IndentedBuffer

    def emit(self, operation: ReductionOpPlan, *, clear: bool) -> None:
        reduce_dims = set(operation.reduce_dims)
        dim_indices = [
            str(index)
            for index, dim in enumerate(operation.source_tensor.dims)
            if dim in reduce_dims
        ]
        dims = f"[{', '.join(dim_indices)}]"
        size = f", size={operation.size}" if operation.size is not None else ""
        self.code.writeline(
            f"T.reduce({operation.source}, {operation.accumulator}, dims={dims}, "
            f"reduce_mode='{operation.reduction_type}', clear={clear!r}{size})"
        )


@dataclasses.dataclass
class BroadcastOpEmitter:
    code: IndentedBuffer

    def emit(self, operation: BroadcastOpPlan) -> None:
        if operation.intrinsic != "vbrc":
            raise RuntimeError(
                f"unsupported TileLang broadcast intrinsic: {operation.intrinsic}"
            )
        self.code.writeline(
            f"T.vbrc({operation.source}, {operation.destination})"
        )


@dataclasses.dataclass
class LoopRegionEmitter:
    code: IndentedBuffer
    plan: TileLangKernelPlan
    scalar_cache: dict[tuple[str, str], str]
    allocations: dict[str, BufferPlan] = dataclasses.field(default_factory=dict)
    register_values: set[str] = dataclasses.field(default_factory=set)

    def _emit_allocations(self, region: KernelRegionPlan) -> None:
        materialized = []
        for allocation in region.allocations:
            if allocation.name in self.register_values:
                continue
            if allocation.scope != "shared":
                raise RuntimeError(
                    f"unsupported TileLang allocation scope: {allocation.scope}"
                )
            self.code.writeline(
                _tilelang_alloc_shared(
                    allocation.name,
                    allocation.shape,
                    allocation.dtype,
                )
            )
            materialized.append(f"{allocation.name}: 1")
        if materialized:
            self.code.writeline("T.annotate_buffer_versions({" + ", ".join(materialized) + "})")

    def _emit_body(self, region: KernelRegionPlan, *, clear: bool) -> None:
        transfer_emitter = TransferEmitter(self.code)
        for transfer in region.input_transfers:
            transfer_emitter.emit(transfer)
        self._emit_operations(region, clear=clear)
        for transfer in region.output_transfers:
            transfer_emitter.emit(transfer)

    def _emit_operations(self, region: KernelRegionPlan, *, clear: bool) -> None:
        for op_id in region.operations:
            operation = self.plan.operation(op_id)
            if isinstance(operation, PointwiseOpPlan):
                from .ascend import AscendPointwiseEmitter

                AscendPointwiseEmitter(self.code, self.allocations).emit(operation)
            elif isinstance(operation, ReductionOpPlan):
                raise TileLangUnsupported("A5 emission", "reduction VF migration is pending")
            elif isinstance(operation, BroadcastOpPlan):
                raise TileLangUnsupported("A5 emission", "materialized broadcast migration is pending")
            else:
                raise RuntimeError(f"unknown TileLang operation plan: {operation!r}")

    def _emit_x_body(self, region: XTileRegionPlan) -> None:
        transfer_emitter = TransferEmitter(self.code)
        self._emit_allocations(region)
        for transfer in region.input_transfers:
            transfer_emitter.emit(transfer)
        self._emit_operations(region, clear=True)
        for subregion in region.subregions:
            self.emit(subregion)
        for transfer in region.output_transfers:
            transfer_emitter.emit(transfer)

    def emit(self, region: KernelRegionPlan) -> None:
        for statement in region.prologue:
            self.code.writeline(statement)

        if isinstance(region, XTileRegionPlan) and region.serial_loop is not None:
            region.serial_loop.emit_serial_header(self.code)
            with self.code.indent():
                for statement in region.body_prologue:
                    self.code.writeline(statement)
                self._emit_x_body(region)
            return

        if isinstance(region, XTileRegionPlan):
            for statement in region.body_prologue:
                self.code.writeline(statement)
            self._emit_x_body(region)
            return

        if isinstance(region, RTileRegionPlan):
            iteration = region.iteration
            if iteration is None:
                raise RuntimeError("RTileRegionPlan requires an iteration plan")
            self._emit_allocations(region)
            if iteration.mode == "NO_LOOP":
                iteration.loop.emit_base(self.code)
                iteration.loop.emit_full_remain(self.code)
                for statement in region.body_prologue:
                    self.code.writeline(statement)
                self._emit_body(region, clear=True)
                return
            if iteration.mode != "SERIAL":
                raise RuntimeError(
                    f"unknown TileLang R-tile iteration mode: {iteration.mode}"
                )
            iteration.loop.emit_base(self.code)
            iteration.loop.emit_first_remain(self.code)
            for statement in region.body_prologue:
                self.code.writeline(statement)
            self._emit_body(region, clear=True)
            iteration.loop.emit_serial_header(self.code, start=1)
            with self.code.indent():
                iteration.loop.emit_serial_base(self.code)
                iteration.loop.emit_serial_remain(self.code)
                for statement in region.body_prologue:
                    self.code.writeline(statement)
                self._emit_body(region, clear=False)
            return

        self._emit_allocations(region)
        for statement in region.body_prologue:
            self.code.writeline(statement)
        self._emit_body(region, clear=True)


class KernelProgramEmitter:
    """Emits a fully planned TileLang kernel without querying V.graph."""

    def emit(self, plan: TileLangKernelPlan) -> str:
        plan.validate()
        code = IndentedBuffer()
        code.writeline("import tilelang.ascend.language as T")
        code.writeline("import math as _math")
        code.writeline("")
        for statement in plan.preamble:
            code.writeline(statement)
        if "_XBLOCK_SUB = X0BLOCK_SUB" in plan.preamble:
            code.writeline(
                "from torch_npu._inductor.tilelang.codegen.ascend import validate_simd_tile_width"
            )
            code.writeline("validate_simd_tile_width(_XBLOCK_SUB)")
        if plan.preamble:
            code.writeline("")
        SignatureEmitter(code).emit(plan.signature)
        with code.indent():
            code.writeline(
                f"with T.Kernel({plan.launch.grid}) as "
                f"{plan.launch.bindings}:"
            )
            with code.indent():
                allocations = {}

                def collect(regions):
                    for region in regions:
                        allocations.update((item.name, item) for item in region.allocations)
                        if isinstance(region, XTileRegionPlan):
                            collect(region.subregions)

                collect(plan.regions)
                boundaries = {
                    name for operation in plan.operations
                    for name in (*operation.inputs, *operation.outputs)
                }
                register_values = {
                    op.out for operation in plan.operations
                    if isinstance(operation, PointwiseOpPlan)
                    for op in operation.compute.operations
                } - boundaries
                region_emitter = LoopRegionEmitter(code, plan, {}, allocations, register_values)
                for region in plan.regions:
                    region_emitter.emit(region)
        return code.getvalue()


# ---------------------------------------------------------------------------
# Op overrides
# ---------------------------------------------------------------------------

def _set_pending(op: str, operands: list, dtype: Optional[torch.dtype] = None) -> None:
    """Set _pending_op on the current TileLangKernel (called from overrides)."""
    try:
        k = V.kernel
        if isinstance(k, TileLangKernel):
            k._pending_op = PendingOp.create(op, operands, dtype)
    except AttributeError:
        pass


class TileLangOverrides(OpOverrides):
    """
    Maps inductor element-wise ops to scalar Python expressions (used for
    the string CSE) and simultaneously records the op type on the kernel
    for NPU vector code emission.
    """

    @staticmethod
    def to_dtype(x, dtype: torch.dtype, src_dtype=None, use_compute_types=True):
        _set_pending("to_dtype", [x], dtype)
        if dtype == torch.bool:
            return f"({x} != 0)"
        return f"T.cast({x}, '{tilelang_dtype(dtype)}')"

    @staticmethod
    def to_dtype_bitcast(x, dtype: torch.dtype, src_dtype: torch.dtype):
        return (
            f"T.reinterpret_cast({x}, '{tilelang_dtype(src_dtype)}', "
            f"'{tilelang_dtype(dtype)}')"
        )

    @staticmethod
    def constant(value, dtype: torch.dtype):
        import torch._prims_common as prim
        literal = repr(prim.dtype_to_type(dtype)(value))
        _set_pending("const", [literal])
        return literal

    @staticmethod
    def abs(x):
        _set_pending("abs", [x])
        return f"abs({x})"

    @staticmethod
    def neg(x):
        _set_pending("neg", [x])
        return f"(-{x})"

    @staticmethod
    def exp(x):
        _set_pending("exp", [x])
        return f"T.exp({x})"

    @staticmethod
    def expm1(x):
        # No single NPU intrinsic; will be caught as unsupported during emit
        return f"(T.exp({x}) - 1.0)"

    @staticmethod
    def log(x):
        _set_pending("log", [x])
        return f"_math.log({x})"

    @staticmethod
    def log1p(x):
        return f"_math.log1p({x})"

    @staticmethod
    def sqrt(x):
        _set_pending("sqrt", [x])
        return f"_math.sqrt({x})"

    @staticmethod
    def rsqrt(x):
        _set_pending("rsqrt", [x])
        return f"(1.0 / _math.sqrt({x}))"

    @staticmethod
    def sin(x):
        _set_pending("sin", [x])
        return f"_math.sin({x})"

    @staticmethod
    def cos(x):
        _set_pending("cos", [x])
        return f"_math.cos({x})"

    @staticmethod
    def tan(x):
        _set_pending("tan", [x])
        return f"_math.tan({x})"

    @staticmethod
    def tanh(x):
        _set_pending("tanh", [x])
        return f"_math.tanh({x})"

    @staticmethod
    def asin(x):
        _set_pending("asin", [x])
        return f"_math.asin({x})"

    @staticmethod
    def acos(x):
        _set_pending("acos", [x])
        return f"_math.acos({x})"

    @staticmethod
    def atan(x):
        _set_pending("atan", [x])
        return f"_math.atan({x})"

    @staticmethod
    def atan2(x, y):
        _set_pending("atan2", [x, y])
        return f"_math.atan2({x}, {y})"

    @staticmethod
    def sigmoid(x):
        _set_pending("sigmoid", [x])
        return f"T.sigmoid({x})"

    @staticmethod
    def relu(x):
        _set_pending("relu", [x])
        return f"(({x}) if ({x}) > 0.0 else 0.0)"

    @staticmethod
    def minimum(a, b):
        _set_pending("minimum", [a, b])
        return f"(({a}) if ({a}) < ({b}) else ({b}))"

    @staticmethod
    def maximum(a, b):
        _set_pending("maximum", [a, b])
        return f"(({a}) if ({a}) > ({b}) else ({b}))"

    @staticmethod
    def where(cond, a, b):
        _set_pending("where", [cond, a, b])
        return f"(({a}) if ({cond}) else ({b}))"

    @staticmethod
    def add(a, b):
        _set_pending("add", [a, b])
        return f"({a} + {b})"

    @staticmethod
    def sub(a, b):
        _set_pending("sub", [a, b])
        return f"({a} - {b})"

    @staticmethod
    def mul(a, b):
        _set_pending("mul", [a, b])
        return f"({a} * {b})"

    @staticmethod
    def truediv(a, b):
        _set_pending("truediv", [a, b])
        return f"({a} / {b})"

    @staticmethod
    def floordiv(a, b):
        _set_pending("floordiv", [a, b])
        return f"({a} // {b})"

    @staticmethod
    def mod(a, b):
        _set_pending("mod", [a, b])
        return f"({a} % {b})"

    @staticmethod
    def logical_not(a):
        _set_pending("logical_not", [a])
        return f"(not ({a}))"

    @staticmethod
    def logical_and(a, b):
        _set_pending("logical_and", [a, b])
        return f"(({a}) and ({b}))"

    @staticmethod
    def logical_or(a, b):
        _set_pending("logical_or", [a, b])
        return f"(({a}) or ({b}))"

    @staticmethod
    def logical_xor(a, b):
        _set_pending("logical_xor", [a, b])
        return f"(bool({a}) != bool({b}))"

    @staticmethod
    def bitwise_and(a, b):
        _set_pending("bitwise_and", [a, b])
        return f"(({a}) & ({b}))"

    @staticmethod
    def bitwise_or(a, b):
        _set_pending("bitwise_or", [a, b])
        return f"(({a}) | ({b}))"

    @staticmethod
    def bitwise_xor(a, b):
        _set_pending("bitwise_xor", [a, b])
        return f"(({a}) ^ ({b}))"

    @staticmethod
    def bitwise_not(a):
        _set_pending("bitwise_not", [a])
        return f"(~({a}))"

    @staticmethod
    def sign(x):  return f"(1 if ({x}) > 0 else (-1 if ({x}) < 0 else 0))"

    @staticmethod
    def floor(x):
        _set_pending("floor", [x])
        return f"_math.floor({x})"

    @staticmethod
    def ceil(x):
        _set_pending("ceil", [x])
        return f"_math.ceil({x})"

    @staticmethod
    def trunc(x):
        _set_pending("trunc", [x])
        return f"_math.trunc({x})"

    @staticmethod
    def erf(x):
        _set_pending("erf", [x])
        return f"_math.erf({x})"

    @staticmethod
    def erfc(x):
        _set_pending("erfc", [x])
        return f"_math.erfc({x})"

    @staticmethod
    def lt(a, b):
        _set_pending("lt", [a, b])
        return f"({a} < {b})"

    @staticmethod
    def le(a, b):
        _set_pending("le", [a, b])
        return f"({a} <= {b})"

    @staticmethod
    def gt(a, b):
        _set_pending("gt", [a, b])
        return f"({a} > {b})"

    @staticmethod
    def ge(a, b):
        _set_pending("ge", [a, b])
        return f"({a} >= {b})"

    @staticmethod
    def eq(a, b):
        _set_pending("eq", [a, b])
        return f"({a} == {b})"

    @staticmethod
    def ne(a, b):
        _set_pending("ne", [a, b])
        return f"({a} != {b})"

    @staticmethod
    def rand(seed, offset):
        raise NotImplementedError("TileLang backend: rand() not supported")

    @staticmethod
    def randint64(seed, offset, low, high):
        raise NotImplementedError("TileLang backend: randint64() not supported")

    @staticmethod
    def load_seed(name, offset):
        raise NotImplementedError("TileLang backend: load_seed() not supported")

    @staticmethod
    def index_expr(expr, dtype):
        return str(V.kernel.rename_indexing(expr))


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

_DEFAULT_XBLOCK = 128
_DEFAULT_REDUCTION_XBLOCK = 8
_DEFAULT_REDUCTION_RBLOCK = 1024
_TILELANG_AUTOTUNE_ENV = "INDUCTOR_ASCEND_TILELANG_AUTOTUNE"
_TILELANG_AUTOTUNE_CANDIDATES_ENV = "INDUCTOR_ASCEND_TILELANG_AUTOTUNE_CANDIDATES"
_TILELANG_CONFIG_GENERATORS_ENV = "INDUCTOR_ASCEND_TILELANG_CONFIG_GENERATORS"
_TILELANG_CARVER_TOPK_ENV = "INDUCTOR_ASCEND_TILELANG_CARVER_TOPK"
_TILELANG_CARVER_CUSTOM_MEM_MUL_ENV = "INDUCTOR_ASCEND_TILELANG_CARVER_CUSTOM_MEM_MUL"
_TILELANG_AUTOTUNE_WARMUP_ENV = "INDUCTOR_ASCEND_TILELANG_AUTOTUNE_WARMUP"
_TILELANG_AUTOTUNE_REP_ENV = "INDUCTOR_ASCEND_TILELANG_AUTOTUNE_REP"
_TILELANG_AUTOTUNE_TIMEOUT_ENV = "INDUCTOR_ASCEND_TILELANG_AUTOTUNE_TIMEOUT"


def _tilelang_env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no"}


def _tilelang_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _tilelang_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _tilelang_autotune_enabled() -> bool:
    return _tilelang_env_flag(_TILELANG_AUTOTUNE_ENV, False)


def _tilelang_autotune_warmup() -> int:
    return max(0, _tilelang_env_int(_TILELANG_AUTOTUNE_WARMUP_ENV, 5))


def _tilelang_autotune_rep() -> int:
    return max(1, _tilelang_env_int(_TILELANG_AUTOTUNE_REP_ENV, 30))


def _tilelang_autotune_timeout() -> int:
    return max(1, _tilelang_env_int(_TILELANG_AUTOTUNE_TIMEOUT_ENV, 30))


def _tilelang_config_generators() -> tuple[str, ...]:
    raw = os.getenv(_TILELANG_CONFIG_GENERATORS_ENV, "tile_generator")
    aliases = {
        "tile_generator": "tile_generator",
        "tile-generator": "tile_generator",
        "tilegen": "tile_generator",
        "generator": "tile_generator",
        "default": "tile_generator",
        "carver": "carver",
    }
    values: list[str] = []
    for item in raw.replace(";", ",").split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item in {"all", "both"}:
            for value in ("tile_generator", "carver"):
                if value not in values:
                    values.append(value)
            continue
        value = aliases.get(item)
        if value is not None and value not in values:
            values.append(value)
    return tuple(values or ("tile_generator",))


def _tilelang_carver_topk() -> int:
    return max(1, _tilelang_env_int(_TILELANG_CARVER_TOPK_ENV, 30))


def _tilelang_carver_custom_mem_mul() -> float:
    return max(0.0, _tilelang_env_float(_TILELANG_CARVER_CUSTOM_MEM_MUL_ENV, 8.0))


def _tilelang_parse_xblock_candidate_filter() -> Optional[set[int]]:
    raw = os.getenv(_TILELANG_AUTOTUNE_CANDIDATES_ENV)
    if raw is None:
        return None
    candidates: set[int] = set()
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError:
            continue
        if value > 0:
            candidates.add(value)
    return candidates or None


def _tilelang_xblock_configs(
    xnumel: int,
    rnumel: int,
    *,
    is_reduction: bool,
    dtype_bytes: int,
    buffer_count: int,
    axis_numels: Optional[list[int]] = None,
    axis_names: Optional[list[str]] = None,
    tiling_axis: Optional[list[int]] = None,
    no_loop_axis: Optional[list[int]] = None,
    split_axis: Optional[list[int]] = None,
    low_dims: Optional[list[int]] = None,
    persistent_reduction: bool = False,
    dual_reduction: bool = False,
) -> list[dict[str, Any]]:
    """Generate TileLang tiling configs for TileLang autotuning."""
    xnumel = max(1, int(xnumel))
    rnumel = max(1, int(rnumel))
    buffer_count = min(max(1, int(buffer_count)), 3)

    dtype = {
        1: torch.int8,
        2: torch.float16,
        4: torch.float32,
        8: torch.int64,
    }.get(max(1, int(dtype_bytes)), torch.float32)
    fallback_hint = _DEFAULT_REDUCTION_XBLOCK if is_reduction else _DEFAULT_XBLOCK
    fallback_rblock = min(rnumel, _DEFAULT_REDUCTION_RBLOCK)

    def normalize_axis_numels(values: Optional[list[int]]) -> list[int]:
        if values is None:
            return [xnumel, rnumel] if is_reduction else [xnumel]
        normalized = [max(1, int(value)) for value in values]
        return normalized or ([xnumel, rnumel] if is_reduction else [xnumel])

    def normalize_axis_names(names: Optional[list[str]], num_axes: int) -> list[str]:
        if names is not None and len(names) == num_axes:
            return [str(name) for name in names]
        if num_axes == 1:
            return ["x0"]
        generated: list[str] = []
        x_count = 0
        r_count = 0
        for idx in range(num_axes):
            if is_reduction and idx == num_axes - 1:
                generated.append(f"r{r_count}")
                r_count += 1
            else:
                generated.append(f"x{x_count}")
                x_count += 1
        return generated

    def normalize_axis_indices(
        values: Optional[list[int]],
        default: list[int],
        num_axes: int,
    ) -> list[int]:
        if values is None:
            values = default
        result: list[int] = []
        for value in values:
            index = int(value)
            if 0 <= index < num_axes and index not in result:
                result.append(index)
        return result

    def first_config_value(
        cfg_kwargs: dict[str, Any],
        prefix: str,
        suffix: str,
    ) -> Optional[int]:
        exact = f"{prefix}0{suffix}"
        if exact in cfg_kwargs:
            return int(cfg_kwargs[exact])
        for key in sorted(cfg_kwargs):
            if key.startswith(prefix) and key.endswith(suffix):
                return int(cfg_kwargs[key])
        return None

    tile_axis_numels = normalize_axis_numels(axis_numels)
    tile_axis_names = normalize_axis_names(axis_names, len(tile_axis_numels))
    tile_split_axis = normalize_axis_indices(
        split_axis,
        [0, 1] if is_reduction and len(tile_axis_numels) > 1 else [0],
        len(tile_axis_numels),
    )
    tile_tiling_axis = normalize_axis_indices(
        tiling_axis,
        [0] if not is_reduction else [],
        len(tile_axis_numels),
    )
    tile_no_loop_axis = normalize_axis_indices(
        no_loop_axis,
        [],
        len(tile_axis_numels),
    )
    tile_low_dims = normalize_axis_indices(
        low_dims,
        [],
        len(tile_axis_numels),
    )

    if not is_reduction and len(tile_axis_numels) > 1:
        ranked_configs: list[dict[str, Any]] = []
        ranked_seen: set[tuple[Any, ...]] = set()
        ranked_block_names = [
            f"{axis_name.upper()}BLOCK" for axis_name in tile_axis_names
        ]

        def add_ranked_config(
            values: Sequence[int],
            multibuffer: bool = False,
        ) -> None:
            blocks = tuple(
                min(axis_numel, max(1, int(value)))
                for axis_numel, value in zip(tile_axis_numels, values)
            )
            if any(
                axis_numel > 1 and block == 1
                for axis_numel, block in zip(tile_axis_numels, blocks)
            ):
                return
            key = (*blocks, bool(multibuffer))
            if key in ranked_seen:
                return
            ranked_seen.add(key)
            config = dict(zip(ranked_block_names, blocks))
            config["multibuffer"] = bool(multibuffer)
            ranked_configs.append(config)

        # Keep the default tile within the historical pointwise element budget,
        # while reserving at least two lanes for every non-singleton outer axis.
        remaining = _DEFAULT_XBLOCK
        fallback_blocks = [1] * len(tile_axis_numels)
        for axis_index in reversed(range(len(tile_axis_numels))):
            outer_non_singletons = sum(
                axis_numel > 1
                for axis_numel in tile_axis_numels[:axis_index]
            )
            reserved = 2 ** outer_non_singletons
            block = min(
                tile_axis_numels[axis_index],
                max(1, remaining // max(1, reserved)),
            )
            if tile_axis_numels[axis_index] > 1:
                block = max(2, block)
            fallback_blocks[axis_index] = block
            remaining = max(1, remaining // block)
        add_ranked_config(fallback_blocks)

        if "tile_generator" in _tilelang_config_generators():
            try:
                tile_generator = TileGenerator(
                    tile_axis_numels,
                    tile_axis_names,
                    tiling_axis=list(range(len(tile_axis_numels))),
                    no_loop_axis=[],
                    split_axis=list(range(len(tile_axis_numels))),
                    low_dims=tile_low_dims,
                    persistent_reduction=False,
                    dtype=dtype,
                    npu_kernel_type=NPUKernelType.SIMD,
                    input_ptr_num=buffer_count,
                )
                generated_configs = tile_generator.descend_split_tiling()
            except Exception:
                generated_configs = []
            for generated_config in generated_configs:
                cfg_kwargs = getattr(generated_config, "kwargs", {})
                effective_blocks: list[int] = []
                for axis_name, axis_numel in zip(
                    tile_axis_names,
                    tile_axis_numels,
                ):
                    prefix = axis_name.upper()
                    effective_blocks.append(int(
                        cfg_kwargs.get(
                            f"{prefix}BLOCK_SUB",
                            cfg_kwargs.get(f"{prefix}BLOCK", axis_numel),
                        )
                    ))
                add_ranked_config(
                    effective_blocks,
                    bool(cfg_kwargs.get("multibuffer", False)),
                )

        return ranked_configs

    reduction_axis_indices = [
        axis_index
        for axis_index, axis_name in enumerate(tile_axis_names)
        if axis_name.startswith("r")
    ]
    pointwise_axis_indices = [
        axis_index
        for axis_index in range(len(tile_axis_names))
        if axis_index not in reduction_axis_indices
    ]
    ranked_reduction = (
        is_reduction
        and len(pointwise_axis_indices) > 1
        and len(reduction_axis_indices) == 1
    )
    if ranked_reduction:
        ranked_configs: list[dict[str, Any]] = []
        ranked_seen: set[tuple[Any, ...]] = set()
        ranked_block_names = [
            f"{axis_name.upper()}BLOCK" for axis_name in tile_axis_names
        ]
        reduction_axis_index = reduction_axis_indices[0]
        reduction_is_no_loop = (
            persistent_reduction
            or reduction_axis_index in tile_no_loop_axis
        )

        def add_ranked_reduction_config(
            values: Sequence[int],
            multibuffer: bool = False,
        ) -> None:
            blocks = [
                min(axis_numel, max(1, int(value)))
                for axis_numel, value in zip(tile_axis_numels, values)
            ]
            if any(
                tile_axis_numels[axis_index] > 1
                and blocks[axis_index] == 1
                for axis_index in pointwise_axis_indices
            ):
                return
            if reduction_is_no_loop:
                blocks[reduction_axis_index] = tile_axis_numels[
                    reduction_axis_index
                ]
            key = (*blocks, bool(multibuffer))
            if key in ranked_seen:
                return
            ranked_seen.add(key)
            config = dict(zip(ranked_block_names, blocks))
            config["multibuffer"] = bool(multibuffer)
            ranked_configs.append(config)

        fallback_blocks = [1] * len(tile_axis_numels)
        for axis_index in pointwise_axis_indices:
            fallback_blocks[axis_index] = min(
                tile_axis_numels[axis_index],
                2,
            )
        pointwise_tile_elements = functools.reduce(
            lambda lhs, rhs: lhs * rhs,
            (fallback_blocks[index] for index in pointwise_axis_indices),
            1,
        )
        fallback_blocks[reduction_axis_index] = (
            tile_axis_numels[reduction_axis_index]
            if reduction_is_no_loop
            else min(
                tile_axis_numels[reduction_axis_index],
                max(
                    1,
                    _DEFAULT_REDUCTION_RBLOCK // pointwise_tile_elements,
                ),
            )
        )
        # A non-singleton block must have at least two lanes: otherwise a
        # logical singleton expansion materializes equal local shapes and the
        # NPU vbrc op receives an empty broadcast-dimension list. Scale RBLOCK
        # down to keep the historical reduction tile element budget.
        add_ranked_reduction_config(fallback_blocks)

        if "tile_generator" in _tilelang_config_generators():
            try:
                tile_generator = TileGenerator(
                    tile_axis_numels,
                    tile_axis_names,
                    tiling_axis=tile_tiling_axis,
                    no_loop_axis=tile_no_loop_axis,
                    split_axis=tile_split_axis,
                    low_dims=tile_low_dims,
                    persistent_reduction=bool(persistent_reduction),
                    dtype=dtype,
                    npu_kernel_type=NPUKernelType.SIMD,
                    input_ptr_num=buffer_count,
                    dual_reduction=bool(dual_reduction),
                )
                generated_configs = tile_generator.descend_split_tiling()
            except Exception:
                generated_configs = []
            for generated_config in generated_configs:
                cfg_kwargs = getattr(generated_config, "kwargs", {})
                effective_blocks: list[int] = []
                valid = True
                for axis_name in tile_axis_names:
                    prefix = axis_name.upper()
                    value = cfg_kwargs.get(
                        f"{prefix}BLOCK_SUB",
                        cfg_kwargs.get(f"{prefix}BLOCK"),
                    )
                    if value is None:
                        valid = False
                        break
                    effective_blocks.append(int(value))
                if valid:
                    add_ranked_reduction_config(
                        effective_blocks,
                        bool(cfg_kwargs.get("multibuffer", False)),
                    )

        return ranked_configs

    reduce_axis_no_loop = bool(
        is_reduction
        and any(
            axis in tile_no_loop_axis
            and tile_axis_names[axis].startswith("r")
            for axis in range(len(tile_axis_names))
        )
    )
    if reduce_axis_no_loop:
        fallback_rblock = rnumel

    allowed = _tilelang_parse_xblock_candidate_filter()
    configs: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, bool]] = set()

    def add_config(
        x0block: int,
        x0block_sub: int,
        r0block: int,
        multibuffer: bool = False,
    ) -> None:
        x0block = max(1, int(x0block))
        x0block_sub = min(max(1, int(x0block_sub)), x0block)
        if reduce_axis_no_loop:
            r0block = rnumel
        r0block = min(max(1, int(r0block)), rnumel)
        if allowed is not None and x0block not in allowed:
            return
        key = (
            x0block,
            x0block_sub,
            r0block if is_reduction else 1,
            bool(multibuffer),
        )
        if key in seen:
            return
        seen.add(key)
        config = {"X0BLOCK": x0block, "X0BLOCK_SUB": x0block_sub}
        if is_reduction:
            config["R0BLOCK"] = r0block
        config["multibuffer"] = bool(multibuffer)
        configs.append(config)

    def add_tile_generator_configs() -> None:
        try:
            tile_generator = TileGenerator(
                tile_axis_numels,
                tile_axis_names,
                tiling_axis=tile_tiling_axis,
                no_loop_axis=tile_no_loop_axis,
                split_axis=tile_split_axis,
                low_dims=tile_low_dims,
                persistent_reduction=bool(persistent_reduction),
                dtype=dtype,
                npu_kernel_type=NPUKernelType.SIMD,
                input_ptr_num=buffer_count,
                dual_reduction=bool(dual_reduction),
            )
            tile_configs = tile_generator.descend_split_tiling()
        except Exception:
            tile_configs = []

        if is_reduction and not tile_configs:
            try:
                tile_generator = TileGenerator(
                    [xnumel],
                    ["x0"],
                    tiling_axis=[],
                    no_loop_axis=[],
                    split_axis=[0],
                    low_dims=[],
                    persistent_reduction=False,
                    dtype=dtype,
                    npu_kernel_type=NPUKernelType.SIMD,
                    input_ptr_num=buffer_count,
                )
                tile_configs = tile_generator.descend_split_tiling()
            except Exception:
                tile_configs = []

        for cfg in tile_configs:
            add_tile_generator_config(cfg)

    def add_tile_generator_config(cfg: Any) -> None:
        cfg_kwargs = getattr(cfg, "kwargs", {})
        x0block = first_config_value(cfg_kwargs, "X", "BLOCK")
        if x0block is None:
            return
        x0block = int(x0block)
        x0block_sub = first_config_value(cfg_kwargs, "X", "BLOCK_SUB")
        x0block_sub = int(x0block_sub if x0block_sub is not None else x0block)
        r0block = first_config_value(cfg_kwargs, "R", "BLOCK")
        if r0block is None:
            r0block = first_config_value(cfg_kwargs, "R", "BLOCK_SUB")
        r0block = int(r0block if r0block is not None else fallback_rblock)
        add_config(
            x0block,
            x0block_sub,
            r0block,
            bool(cfg_kwargs.get("multibuffer", False)),
        )

    def add_carver_configs() -> None:
        try:
            from tilelang import carver
            from tilelang.utils.npu_arch import AscendArch
        except Exception:
            return

        try:
            custom_mem_mul = _tilelang_carver_custom_mem_mul()
            if is_reduction:
                structure = "".join(
                    "R" if name.startswith("r") else "S"
                    for name in tile_axis_names
                )
                if "R" not in structure or "S" not in structure:
                    structure = "SR"
                    shape = [xnumel, rnumel]
                else:
                    shape = tile_axis_numels
                carver_template = carver.GeneralReductionTemplate(
                    structure=structure,
                    shape=list(shape),
                    dtype=tilelang_dtype(dtype),
                    custom_mem_mul=custom_mem_mul,
                ).with_arch(AscendArch())
            else:
                carver_template = carver.ElementwiseTemplate(
                    shape=[xnumel],
                    dtype=tilelang_dtype(dtype),
                    custom_mem_mul=custom_mem_mul,
                ).with_arch(AscendArch())

            hints = carver_template.recommend_hints(topk=_tilelang_carver_topk())
        except Exception:
            return

        for hint in hints:
            block = [int(value) for value in getattr(hint, "block", [])]
            if not block:
                continue
            x0block = functools.reduce(lambda x, y: x * y, block)
            if is_reduction:
                rstep = [int(value) for value in getattr(hint, "rstep", [])]
                if not rstep:
                    continue
                r0block = functools.reduce(lambda x, y: x * y, rstep)
            else:
                r0block = fallback_rblock
            add_config(x0block, x0block, r0block)

    # The first config is also the non-autotune fallback. Keep it bounded by
    # the conservative defaults instead of inheriting TileGenerator's largest
    # candidate (for example R0BLOCK=8192), which can overflow UB before the
    # autotuner has a chance to reject it.
    if allowed is None or fallback_hint in allowed:
        add_config(fallback_hint, fallback_hint, fallback_rblock)

    for generator in _tilelang_config_generators():
        if generator == "tile_generator":
            add_tile_generator_configs()
        elif generator == "carver":
            add_carver_configs()

    if not configs:
        allowed = None
        add_config(fallback_hint, fallback_hint, fallback_rblock)
    return configs

def _tilelang_ascend_pass_configs(
    multi_buffer: Optional[bool] = None,
) -> dict[str, Any]:
    if multi_buffer:
        raise NotImplementedError("A5 requires a per-buffer version plan; boolean multibuffer is unsupported")
    return {"tl.enable_auto_schedule": True}


_SUPPORTED_REDUCTIONS = frozenset({"sum", "max", "min"})


_MAX_REDUCTIONS_PER_KERNEL = 2


def _resolve_operand(operand) -> str:
    """Convert a CSEVariable or string operand to a string suitable for T.v* calls."""
    if isinstance(operand, (int, float)):
        return repr(operand)
    return str(operand)


def _is_scalar_literal_operand(operand: Any) -> bool:
    return isinstance(operand, (int, float)) and not isinstance(operand, bool)


def _tilelang_expr_key(
    var,
    values: dict[str, TileLangValueInfo],
    _memo: Optional[dict[str, tuple]] = None,
) -> tuple:
    """Build a stable key for reusing already materialized TileLang expressions."""
    if _memo is None:
        _memo = {}

    var_name = str(var)
    if var_name in _memo:
        return _memo[var_name]
    value_info = values.get(var_name)
    if value_info is not None and value_info.buffer is not None:
        key = ("buf", value_info.buffer)
    elif value_info is not None and value_info.const is not None:
        key = ("const", str(value_info.const))
    elif value_info is not None and value_info.op is not None:
        op_name = value_info.op.name
        operands = value_info.op.operands
        key = (
            "op",
            op_name,
            value_info.dtype,
            tuple(
                _tilelang_expr_key(op, values, _memo)
                for op in operands
            ),
        )
    else:
        key = ("literal", var_name)
    _memo[var_name] = key
    return key


def _finalize_vec_ops_output(
    ops_list: list[VecOp],
    out_buf: str,
) -> None:
    if not ops_list:
        return
    # The execution dtype is normalized later from materialized tensor operands.
    # Explicit casts are represented as "copy" ops by to_dtype and keep their
    # own dtype.
    ops_list[-1] = ops_list[-1].with_out(out_buf)


def _select_vec_op_execution_dtype(
    spec: TileLangOpSpec,
    operand_dtypes: Sequence[Optional[torch.dtype]],
    out_dtype: Optional[torch.dtype],
    record_dtype: Optional[torch.dtype],
    default_dtype: torch.dtype,
) -> torch.dtype:
    if spec.name == "copy":
        source_dtype = operand_dtypes[0] if operand_dtypes else None
        return record_dtype or out_dtype or source_dtype or default_dtype

    candidates = [
        dtype
        for index, dtype in enumerate(operand_dtypes)
        if dtype is not None and index not in spec.predicate_operands
    ]

    if candidates:
        first_dtype = candidates[0]
        if all(operand_dtype == first_dtype for operand_dtype in candidates):
            return first_dtype
        if out_dtype in candidates:
            return out_dtype
        if record_dtype in candidates:
            return record_dtype
        if default_dtype in candidates:
            return default_dtype
        return first_dtype
    if out_dtype is not None:
        return out_dtype
    if record_dtype is not None:
        return record_dtype
    return default_dtype


def _normalize_vec_op_dtypes(
    ops_list: list[VecOp],
    buffer_dtypes: dict[str, torch.dtype],
    default_dtype: torch.dtype,
) -> list[VecOp]:
    logical_dtypes = dict(buffer_dtypes)
    normalized_ops: list[VecOp] = []
    for op_record in ops_list:
        operand_dtypes = [
            logical_dtypes.get(operand) if isinstance(operand, str) else None
            for operand in op_record.operands
        ]
        out_dtype = logical_dtypes.get(op_record.out)
        execution_dtype = _select_vec_op_execution_dtype(
            op_record.spec,
            operand_dtypes,
            out_dtype,
            op_record.dtype,
            default_dtype,
        )
        result_dtype = op_record.spec.result_dtype(
            execution_dtype,
            op_record.dtype,
            out_dtype,
        )
        normalized_op = op_record.with_dtypes(execution_dtype, result_dtype)
        normalized_ops.append(normalized_op)
        logical_dtypes[op_record.out] = result_dtype
    return normalized_ops


def _build_vec_ops(
    var,
    target_buf: str,
    ops_list: list[VecOp],
    values: dict[str, TileLangValueInfo],
    reusable_expr_bufs: Optional[dict[tuple, str]] = None,
    _visited: Optional[set] = None,
    reusable_expr_predicate: Optional[Callable[[str], bool]] = None,
) -> str:
    """
    Recursively traverse the op graph rooted at `var` and append
    VecOp records to `ops_list` in
    topological (post) order.

    Returns the buffer name that will hold the result of `var`.
    For input-buffer vars this is the existing local buffer name.
    For computed vars this is `target_buf`.
    """
    if _visited is None:
        _visited = set()

    var_name = str(var)
    value_info = values.get(var_name)

    # Direct input buffer
    if value_info is not None and value_info.buffer is not None:
        return value_info.buffer

    # Constant literal that went through CSE (e.g. tmp1 = 2.0)
    if value_info is not None and value_info.const is not None:
        try:
            return float(value_info.const)
        except (ValueError, TypeError):
            return value_info.const

    # Computed var
    if value_info is not None and value_info.op is not None:
        cacheable_expr = (
            reusable_expr_bufs is not None
            and (
                reusable_expr_predicate is None
                or reusable_expr_predicate(var_name)
            )
        )
        expr_key = None
        if cacheable_expr:
            expr_key = _tilelang_expr_key(var, values)
            if expr_key in reusable_expr_bufs:
                return reusable_expr_bufs[expr_key]

        if var_name in _visited:
            return f"_{var_name}_frag"

        _visited.add(var_name)
        op_name = value_info.op.name
        operands = value_info.op.operands

        if op_name == "to_dtype":
            if len(operands) != 1:
                raise RuntimeError("TileLang to_dtype expects one operand")
            op = operands[0]
            op_str = str(op)
            op_info = values.get(op_str)
            if op_info is not None and op_info.buffer is not None:
                src = op_info.buffer
            elif op_info is not None and op_info.const is not None:
                try:
                    src = float(op_info.const)
                except (ValueError, TypeError):
                    src = op_info.const
            elif op_info is not None and op_info.op is not None:
                inter_buf = f"_{op_str}_frag"
                src = _build_vec_ops(
                    op,
                    inter_buf,
                    ops_list,
                    values,
                    reusable_expr_bufs,
                    _visited,
                    reusable_expr_predicate,
                )
            else:
                try:
                    src = float(op_str)
                except (ValueError, TypeError):
                    src = op_str

            src_dtype = op_info.dtype if op_info is not None else None
            dst_dtype = value_info.dtype
            if src == target_buf or (src_dtype is not None and src_dtype == dst_dtype):
                if cacheable_expr and expr_key is not None:
                    reusable_expr_bufs[expr_key] = src
                return src
            ops_list.append(VecOp.create("copy", (src,), target_buf, dst_dtype))
            if cacheable_expr and expr_key is not None:
                reusable_expr_bufs[expr_key] = target_buf
            return target_buf

        resolved = []
        for op in operands:
            op_str = str(op)
            op_info = values.get(op_str)
            if op_info is not None and op_info.buffer is not None:
                resolved.append(op_info.buffer)
            elif op_info is not None and op_info.const is not None:
                try:
                    resolved.append(float(op_info.const))
                except (ValueError, TypeError):
                    resolved.append(op_info.const)
            elif op_info is not None and op_info.op is not None:
                inter_buf = f"_{op_str}_frag"
                src = _build_vec_ops(
                    op,
                    inter_buf,
                    ops_list,
                    values,
                    reusable_expr_bufs,
                    _visited,
                    reusable_expr_predicate,
                )
                resolved.append(src)
            else:
                # Raw literal string (e.g. "2.0" passed directly without going through CSE)
                try:
                    resolved.append(float(op_str))
                except (ValueError, TypeError):
                    resolved.append(op_str)
        ops_list.append(
            VecOp.create(op_name, resolved, target_buf, value_info.dtype)
        )
        if cacheable_expr and expr_key is not None:
            reusable_expr_bufs[expr_key] = target_buf
        return target_buf

    # Fallback: treat as a literal / unknown symbol
    return var_name


def _is_reduction_symbol_name(name: str) -> bool:
    return name == "r" or name == "rindex" or re.match(r"r\d+(?:_|$)", name) is not None


def _is_pointwise_symbol_name(name: str) -> bool:
    return bool(name) and name[0] in "wvtzyx"


def _has_reduction_index(index: sympy.Expr) -> bool:
    return any(
        _is_reduction_symbol_name(getattr(symbol, "name", str(symbol)))
        for symbol in index.free_symbols
    )


def _has_pointwise_index(index: sympy.Expr) -> bool:
    return any(
        _is_pointwise_symbol_name(getattr(symbol, "name", str(symbol)))
        for symbol in index.free_symbols
    )


def _dense_reduction_matrix_expected_index(
    index: sympy.Expr,
    reduction_numel: sympy.Expr,
) -> Optional[sympy.Expr]:
    pointwise_symbols = [
        symbol for symbol in index.free_symbols
        if _is_pointwise_symbol_name(getattr(symbol, "name", str(symbol)))
    ]
    r_symbols = [
        symbol for symbol in index.free_symbols
        if _is_reduction_symbol_name(getattr(symbol, "name", str(symbol)))
    ]
    if len(pointwise_symbols) != 1 or len(r_symbols) != 1:
        return None
    return pointwise_symbols[0] * reduction_numel + r_symbols[0]


def _is_dense_reduction_matrix_index(
    index: sympy.Expr,
    reduction_numel: sympy.Expr,
) -> bool:
    expected = _dense_reduction_matrix_expected_index(index, reduction_numel)
    if expected is None:
        return False
    try:
        return bool(sympy.simplify(index - expected) == 0)
    except Exception:
        return index == expected


def _tilelang_index_expr(
    expr: sympy.Expr,
    pointwise_index: str = "cid",
    reduction_index: str = "_tl_i",
    symbol_replacements: Optional[dict[sympy.Symbol, str]] = None,
) -> str:
    if symbol_replacements is None:
        symbol_replacements = {}
    if isinstance(expr, (int, sympy.Integer)):
        return str(int(expr))
    if isinstance(expr, sympy.Symbol):
        if expr in symbol_replacements:
            return symbol_replacements[expr]
        name = getattr(expr, "name", str(expr))
        if _is_pointwise_symbol_name(name):
            return pointwise_index
        if _is_reduction_symbol_name(name):
            return reduction_index
        return name
    if isinstance(expr, FloorDiv):
        return (
            f"({_tilelang_index_expr(expr.args[0], pointwise_index, reduction_index, symbol_replacements)} "
            f"// {_tilelang_index_expr(expr.args[1], pointwise_index, reduction_index, symbol_replacements)})"
        )
    if isinstance(expr, ModularIndexing):
        base, divisor, modulus = expr.args
        base_expr = _tilelang_index_expr(
            base,
            pointwise_index,
            reduction_index,
            symbol_replacements,
        )
        if divisor != 1:
            base_expr = (
                f"({base_expr} // "
                f"{_tilelang_index_expr(divisor, pointwise_index, reduction_index, symbol_replacements)})"
            )
        return (
            f"({base_expr} % "
            f"{_tilelang_index_expr(modulus, pointwise_index, reduction_index, symbol_replacements)})"
        )
    if isinstance(expr, sympy.Add):
        return "(" + " + ".join(
            _tilelang_index_expr(
                arg,
                pointwise_index,
                reduction_index,
                symbol_replacements,
            )
            for arg in expr.args
        ) + ")"
    if isinstance(expr, sympy.Mul):
        return "(" + " * ".join(
            _tilelang_index_expr(
                arg,
                pointwise_index,
                reduction_index,
                symbol_replacements,
            )
            for arg in expr.args
        ) + ")"
    return str(expr)


def _is_zero_index(index: sympy.Expr) -> bool:
    try:
        return bool(sympy.simplify(index) == 0)
    except Exception:
        return False


class TileLangKernel(NPUIndexTritonKernel):
    """Capture typed operation and loop plans for the Ascend pointwise emitter."""

    overrides = TileLangOverrides

    def __init__(self, tiling: dict, **kwargs) -> None:
        super().__init__(tiling, **kwargs)
        self.cse: CSE = CSE(self.newvar_prefix, self.suffix)

        self._tl_static_axis_vars: dict[sympy.Symbol, tuple[str, sympy.Expr]] = {}
        self._tl_static_value_vars: dict[tuple[str, str], tuple[str, sympy.Expr]] = {}
        self._tl_pointwise_axis_lengths: Optional[list[sympy.Expr]] = None
        self._tl_pointwise_axis_names: Optional[list[str]] = None
        self._tl_reduction_dim_names: dict[str, str] = {}
        self._tl_reduction_block: Optional[TileLangBlock] = None
        self._tl_for_loops: dict[str, TileLangForLoop] = {
            "XBLOCK": TileLangForLoop(
                name="XBLOCK",
                extent="_xnumel",
                block="_XBLOCK",
                loop_var="cid",
                base="_tl_x_base",
                remain="_tl_outer_remain",
                base_expr="cid * _XBLOCK",
            ),
            "X0BLOCK": TileLangForLoop(
                name="X0BLOCK",
                extent="_tl_outer_remain",
                block="_XBLOCK_SUB",
                loop_var="_tl_block",
                base="_tl_base",
                remain="_remain_X",
                base_expr="cid * _XBLOCK + _tl_block * _XBLOCK_SUB",
            ),
            "RBLOCK": TileLangForLoop(
                name="RBLOCK",
                extent="_rnumel",
                block="_RBLOCK",
                loop_var="_tl_r",
                base="_r_base",
                remain="_remain_R",
            ),
            "Y0BLOCK": TileLangForLoop(
                name="Y0BLOCK",
                extent="_ynumel",
                block="_YBLOCK",
                loop_var="_tl_y",
                base="_y_base",
                remain="_remain_Y",
            ),
        }

        # Op graph (built during load/store/overrides calls)
        self._pending_op: Optional[PendingOp] = None
        self._values: dict[str, TileLangValueInfo] = {}
        self._buffers: dict[str, TileLangBufferInfo] = {}
        self._outputs: dict[str, TileLangOutputBinding] = {}
        self._reduction_vars: dict[str, tuple[str, TileLangCSEVariable, torch.dtype]] = {}
        self._reduction_outputs: dict[str, tuple[TileLangCSEVariable, torch.dtype]] = {}

    # ------------------------------------------------------------------
    # SIMDKernel abstract interface
    # ------------------------------------------------------------------

    def want_no_x_dim(self) -> bool:
        return False

    def initialize_range_tree(self, pid_cache: Optional[dict[str, str]]) -> None:
        """
        Use NPUIndex range-tree entries so NPUTritonScheduling's indexing
        transform can remove/substitute axes, while keeping TileLang's existing
        x/r prefix convention.
        """
        if pid_cache is None:
            pid_cache = {}

        active_prefixes = [
            prefix for prefix in ("w", "v", "t", "z", "y", "x", "r")
            if prefix in self.numels
        ]
        no_r_dim = not self.inside_reduction or not self.features.is_reduction()

        if self.no_x_dim:
            tensor_dims = ["r"]
        elif no_r_dim:
            tensor_dims = ["x", "y", "z", "t", "v", "w"]
        else:
            tensor_dims = ["x", "y", "z", "t", "v", "w", "r"]

        grid_dims = ["x", "y", "z"]
        tensor_dim_map = {
            prefix: idx
            for idx, prefix in enumerate(p for p in tensor_dims if p in active_prefixes)
        }
        grid_dim_map = {
            prefix: idx
            for idx, prefix in enumerate(p for p in grid_dims if p in active_prefixes)
        }

        for i, prefix in enumerate(active_prefixes):
            is_reduction = prefix.startswith("r")
            tensor_dim = tensor_dim_map.get(prefix)
            grid_dim = None if is_reduction else grid_dim_map.get(prefix)
            index = i if grid_dim is None else grid_dim
            self.range_trees.append(
                IterationRangesRootNPUIndex(
                    f"{prefix}index",
                    self.numels[prefix],
                    prefix,
                    index,
                    self,
                    pid_cache=pid_cache,
                    is_loop=is_reduction and not self.persistent_reduction,
                    tensor_dim=tensor_dim,
                    grid_dim=grid_dim,
                )
            )

    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry) -> None:
        pass

    def iteration_ranges_get_pid(self, entry: IterationRangesRoot) -> str:
        return "cid"

    def iteration_ranges_ranges_code(self, entry: IterationRangesRoot) -> str:
        return f"T.arange(0, {entry.prefix.upper()}BLOCK)"

    def iteration_ranges_scalar_code(self, entry: IterationRangesRoot, value: Any) -> str:
        return repr(value)

    def split_and_set_ranges(self, lengths: Sequence[Sequence[sympy.Expr]]):
        groups = {rt.prefix: rt.numel for rt in self.range_trees}
        if not self.inside_reduction:
            for prefix in groups:
                if prefix.startswith("r"):
                    groups[prefix] = sympy.S.One
        return self.map_kernel_groups_to_node_sizes(
            [*groups.values()], lengths, self.set_ranges
        )

    def _tilelang_for_loop(self, name: str) -> TileLangForLoop:
        return self._tl_for_loops[name]

    def _linear_pointwise_block(self) -> TileLangBlock:
        return TileLangBlock.create("pointwise_linear", {"x": "_XBLOCK_SUB"})

    def _reduction_tile_block(self) -> TileLangBlock:
        if self._tl_reduction_block is None:
            self._tl_reduction_block = TileLangBlock.create(
                "reduction_tile",
                {"x": "_XBLOCK_SUB", "r": "_RBLOCK"},
            )
        return self._tl_reduction_block

    def _clear_pending_compute_op(self, reason: str) -> None:
        # A load CSE var is a buffer reference, not the scalar expression that
        # may have set _pending_op immediately before this call.
        self._pending_op = None

    def _consume_pending_compute_op(self) -> Optional[PendingOp]:
        pending_op = self._pending_op
        self._pending_op = None
        return pending_op

    def _value_info(self, name: str) -> TileLangValueInfo:
        return self._values.setdefault(name, TileLangValueInfo())

    def _record_value_dtype(self, name: str, dtype: torch.dtype) -> None:
        self._value_info(name).dtype = dtype

    def _record_value_buffer(
        self,
        name: str,
        local: str,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        value_info = self._value_info(name)
        value_info.buffer = local
        if dtype is not None:
            value_info.dtype = dtype

    def _record_value_op(self, name: str, op: PendingOp) -> None:
        value_info = self._value_info(name)
        value_info.op = op
        if op.dtype is not None and value_info.dtype is None:
            value_info.dtype = op.dtype

    def _record_value_const(self, name: str, const: Any) -> None:
        self._value_info(name).const = const

    def _record_buffer(
        self,
        name: str,
        var: str,
        local: str,
        dtype: torch.dtype,
        role: str,
        index: Optional[sympy.Expr] = None,
        numel: Optional[sympy.Expr] = None,
    ) -> TileLangBufferInfo:
        if role == "output":
            for existing_local, existing_info in list(self._buffers.items()):
                if existing_info.role == "output" and existing_info.name == name:
                    del self._buffers[existing_local]
        buffer_info = TileLangBufferInfo(
            name=name,
            var=var,
            local=local,
            dtype=dtype,
            role=role,
            index=index,
            numel=numel,
        )
        self._buffers[local] = buffer_info
        return buffer_info

    def _record_output_binding(
        self,
        local: str,
        value: TileLangCSEVariable,
        dtype: torch.dtype,
        index: Optional[sympy.Expr] = None,
    ) -> None:
        self._outputs[local] = TileLangOutputBinding(local, value, dtype, index)

    def _output_items(
        self,
    ) -> list[tuple[str, tuple[TileLangCSEVariable, torch.dtype]]]:
        return [
            (local, (binding.value, binding.dtype))
            for local, binding in self._outputs.items()
        ]

    def _output_index(self, local: str) -> sympy.Expr:
        binding = self._outputs.get(local)
        if binding is None or binding.index is None:
            return sympy.S.Zero
        return binding.index

    def _pointwise_store_indices(self) -> list[sympy.Expr]:
        return [
            binding.index
            for binding in self._outputs.values()
            if binding.index is not None
        ]

    def _pointwise_store_symbol_set(self) -> set[sympy.Symbol]:
        symbols: set[sympy.Symbol] = set()
        for index in self._pointwise_store_indices():
            symbols.update(index.free_symbols)
        return symbols

    def _representative_pointwise_store_index(self) -> Optional[sympy.Expr]:
        output_indices = self._pointwise_store_indices()
        return output_indices[0] if output_indices else None

    def _buffer_items(
        self,
        role: str,
    ) -> list[tuple[str, tuple[str, str, torch.dtype]]]:
        return [
            (buffer_info.name, (buffer_info.var, buffer_info.local, buffer_info.dtype))
            for buffer_info in self._buffers.values()
            if buffer_info.role == role
        ]

    def _input_items(self) -> list[tuple[str, tuple[str, str, torch.dtype]]]:
        return self._buffer_items("input")

    def _output_buffer_items(self) -> list[tuple[str, tuple[str, str, torch.dtype]]]:
        return self._buffer_items("output")

    def _input_index(self, local: str) -> sympy.Expr:
        buffer_info = self._buffers.get(local)
        if buffer_info is None or buffer_info.index is None:
            return sympy.S.Zero
        return buffer_info.index

    def _input_numel(self, local: str) -> Optional[sympy.Expr]:
        buffer_info = self._buffers.get(local)
        if buffer_info is None or buffer_info.numel is None:
            return None
        return buffer_info.numel

    def _has_output_buffer(self, name: str) -> bool:
        return any(
            buffer_info.name == name and buffer_info.role == "output"
            for buffer_info in self._buffers.values()
        )

    # ------------------------------------------------------------------
    # load / store / reduction
    # ------------------------------------------------------------------

    def load(self, name: str, index: sympy.Expr) -> TileLangCSEVariable:
        if name in self.cse.store_cache:
            return self.cse.store_cache[name]

        dtype = V.graph.get_dtype(name)
        if dtype not in _ANY_SUPPORTED_DTYPE:
            raise NotImplementedError(
                f"TileLang backend: no T.v* op supports dtype {dtype} "
                f"(buffer '{name}'); falling back to Triton."
            )
        var = self.args.input(name)
        local_name = f"_{var}_local"
        input_numel = None
        try:
            input_numel = sympy_product(V.graph.get_buffer(name).get_size())
        except Exception:
            pass
        self._record_buffer(
            name,
            var,
            local_name,
            dtype,
            role="input",
            index=index,
            numel=input_numel,
        )
        self._clear_pending_compute_op("load")
        cse_var = self.cse.generate(self.loads, f"{local_name}[_tl_i]", dtype=dtype)
        self._record_value_buffer(cse_var.name, local_name, dtype)
        return cse_var

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: TileLangCSEVariable,
        mode: StoreMode = None,
    ) -> None:
        if mode == "atomic_add":
            raise NotImplementedError(
                "TileLang backend: atomic_add store not yet supported"
            )
        dtype = V.graph.get_dtype(name)
        var = self.args.output(name)
        local_name = f"_{var}_local"
        if not self._has_output_buffer(name):
            self._record_buffer(name, var, local_name, dtype, role="output")
        # Check that every op in the compute graph supports this output dtype.
        # This is the right place to check: all loads/overrides have already run,
        # so _values already contains the complete op graph. NotImplementedError here
        # propagates through _body() and triggers inductor's Triton fallback.
        self._check_op_graph_dtype(str(value), dtype)
        self._record_output_binding(local_name, value, dtype, index)

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: TileLangCSEVariable,
    ) -> TileLangCSEVariable:
        if not self.inside_reduction:
            raise RuntimeError("TileLang reduction must run inside reduction context")
        if isinstance(value, tuple):
            raise NotImplementedError(
                "TileLang backend: tuple reductions are not yet supported; "
                "falling back to Triton."
            )
        if reduction_type not in _SUPPORTED_REDUCTIONS:
            raise NotImplementedError(
                f"TileLang backend: reduction '{reduction_type}' is not yet "
                "supported; falling back to Triton."
            )
        if len(self._reduction_outputs) >= _MAX_REDUCTIONS_PER_KERNEL:
            raise NotImplementedError(
                "TileLang backend: too many reductions in one kernel "
                f"(max {_MAX_REDUCTIONS_PER_KERNEL}); falling back to Triton."
            )
        existing_reduction_sequence = [
            reduce_info[0] for reduce_info in self._reduction_vars.values()
        ]
        next_reduction_sequence = [*existing_reduction_sequence, reduction_type]
        if len(next_reduction_sequence) > 1 and next_reduction_sequence not in (
            ["max", "sum"],
            ["sum", "sum"],
        ):
            raise NotImplementedError(
                "TileLang backend: multiple reductions currently support only "
                "max+sum and sum+sum patterns; falling back to Triton."
            )
        self._check_op_graph_dtype(str(value), dtype)

        result = self.cse.generate(
            self.compute,
            f"tl_reduce_{reduction_type}({value})",
            dtype=dtype,
        )
        self._reduction_vars[result.name] = (reduction_type, value, dtype)
        return result

    def store_reduction(
        self,
        name: str,
        index: sympy.Expr,
        value: TileLangCSEVariable,
    ) -> None:
        if not self.inside_reduction:
            raise RuntimeError("TileLang store_reduction must run inside reduction context")
        dtype = V.graph.get_dtype(name)
        if str(value) not in self._reduction_vars:
            raise NotImplementedError(
                "TileLang backend: store_reduction expects a direct reduction "
                "result; falling back to Triton."
            )
        local_name = f"_{value}_reduce_local"
        self._record_value_buffer(str(value), local_name, dtype)
        self._reduction_outputs[local_name] = (value, dtype)
        if name not in V.graph.removed_buffers:
            var = self.args.output(name)
            # Unconditional overwrite: store() may run first (e.g. fused cast
            # epilogue) and register the wrong _out_ptr*_local for this name.
            # The reduction buffer must always win.
            self._record_buffer(name, var, local_name, dtype, role="output")

    # ------------------------------------------------------------------
    # Source generation
    # ------------------------------------------------------------------

    def _pointwise_input_kind(self, loc: str) -> str:
        index = self._input_index(loc)
        if _is_zero_index(index):
            return "scalar"
        output_symbols = self._pointwise_store_symbol_set()
        if output_symbols:
            load_symbols = set(index.free_symbols)
            if not output_symbols.issubset(load_symbols):
                return "broadcast"
        return "vector"

    def _pointwise_broadcast_factor(self, loc: str) -> Optional[sympy.Expr]:
        index = self._input_index(loc)
        output_index = self._representative_pointwise_store_index()
        if (
            output_index is None
            or _is_zero_index(index)
            or not isinstance(index, sympy.Symbol)
        ):
            return None

        coeff = sympy.expand(output_index).coeff(index)
        if coeff in (0, 1):
            return None
        return sympy.simplify(coeff)

    def _pointwise_broadcast_shape(self, loc: str) -> str:
        input_numel = self._input_numel(loc)
        if input_numel is not None:
            return f"({self._static_value_var('B', input_numel)},)"
        index = self._input_index(loc)
        modular_terms = list(index.atoms(ModularIndexing))
        if len(modular_terms) == 1:
            return f"({self._static_value_var('B', modular_terms[0].args[2])},)"
        factor = self._pointwise_broadcast_factor(loc)
        if factor is None:
            return "(T.ceildiv(_xnumel, _XBLOCK),)"
        return f"(T.ceildiv(_xnumel, {self._static_value_var('B', factor)}),)"

    def _pointwise_broadcast_lane_index(
        self,
        loc: str,
        pointwise_index: str = "(cid * _XBLOCK + _tl_i)",
    ) -> str:
        factor = self._pointwise_broadcast_factor(loc)
        if factor is not None:
            factor_var = self._static_value_var("B", factor)
            index = f"({pointwise_index} // {factor_var})"
            input_numel = self._input_numel(loc)
            if input_numel is not None:
                numel_var = self._static_value_var("B", input_numel)
                index = f"({index} % {numel_var})"
            return index
        index = self._input_index(loc)
        index_expr = _tilelang_index_expr(
            index,
            pointwise_index=pointwise_index,
        )
        input_numel = self._input_numel(loc)
        if input_numel is not None:
            numel_var = self._static_value_var("B", input_numel)
            index_expr = f"({index_expr} % {numel_var})"
        return index_expr

    def _axis_dim_var(self, axis, extent: sympy.Expr) -> str:
        prefix = getattr(axis, "prefix", "")
        if prefix in self.numels and V.graph.sizevars.statically_known_equals(
            self.numels[prefix],
            extent,
        ):
            return f"_{prefix}numel"
        return self._static_value_var("B", extent)

    def _pointwise_layout_axes(self) -> list:
        if not self.golden_var_list:
            try:
                self.select_golden_varlist()
            except Exception:
                pass

        axes = []
        for symbol in reversed(list(self.golden_var_list or [])):
            axis = (
                self.range_tree_nodes.get(symbol)
                or self.range_tree_nodes_removed.get(symbol)
            )
            if axis is not None and _is_pointwise_symbol_name(axis.name):
                axes.append(axis)
        sorted_axes = [
            axis for axis in getattr(self, "sorted_axis", [])
            if _is_pointwise_symbol_name(axis.name)
        ]
        if len(sorted_axes) > len(axes):
            return sorted_axes
        if axes:
            return axes
        return sorted_axes

    def _pointwise_output_axis_groups(
        self,
        axes: Sequence[Any],
        output_shape: Sequence[sympy.Expr],
    ) -> Optional[list[tuple[int, ...]]]:
        axis_lengths = [getattr(axis, "length", sympy.S.One) for axis in axes]
        shape = list(output_shape)

        def matches(actual: sympy.Expr, expected: sympy.Expr) -> bool:
            return V.graph.sizevars.statically_known_equals(actual, expected)

        def search(axis_index: int, dim_index: int) -> Optional[list[tuple[int, ...]]]:
            if axis_index == len(axis_lengths):
                return [] if dim_index == len(shape) else None

            for end in range(dim_index + 1, len(shape) + 1):
                group = tuple(range(dim_index, end))
                product = sympy_product([shape[i] for i in group])
                if not matches(product, axis_lengths[axis_index]):
                    continue
                rest = search(axis_index + 1, end)
                if rest is not None:
                    return [group, *rest]
            if matches(sympy.S.One, axis_lengths[axis_index]):
                rest = search(axis_index + 1, dim_index)
                if rest is not None:
                    return [(), *rest]
            return None

        return search(0, 0)

    @staticmethod
    def _pointwise_logical_axis_names(rank: int) -> list[str]:
        prefixes = ("w", "v", "t", "z", "y", "x")[-rank:]
        return [f"{prefix}{i}" for i, prefix in enumerate(prefixes)]

    def _pointwise_ranked_broadcast_info(self) -> Optional[PointwiseLayoutPlan]:
        self._tl_pointwise_axis_lengths = None
        self._tl_pointwise_axis_names = None

        if self._reduction_outputs or not self._output_buffer_items():
            return None

        base_axes = self._pointwise_layout_axes()
        if len(base_axes) <= 1:
            return None

        def matches(actual: sympy.Expr, expected: sympy.Expr) -> bool:
            return V.graph.sizevars.statically_known_equals(actual, expected)

        def shape_matches(
            actual: Sequence[sympy.Expr],
            expected: Sequence[sympy.Expr],
        ) -> bool:
            return (
                len(actual) == len(expected)
                and all(matches(a, e) for a, e in zip(actual, expected))
            )

        output_shapes: list[Sequence[sympy.Expr]] = []
        output_dtypes: dict[str, torch.dtype] = {}
        for name, (var, _, dtype) in self._output_buffer_items():
            try:
                shape = V.graph.get_buffer(name).get_size()
            except Exception:
                return None
            output_shapes.append(shape)
            output_dtypes[var] = dtype

        output_shape = output_shapes[0]
        for shape in output_shapes[1:]:
            if not shape_matches(shape, output_shape):
                return None

        base_axis_groups = self._pointwise_output_axis_groups(base_axes, output_shape)
        if base_axis_groups is None:
            base_axis_lengths_from_axes = [
                getattr(axis, "length", sympy.S.One) for axis in base_axes
            ]
            if not V.graph.sizevars.statically_known_equals(
                sympy_product(output_shape),
                sympy_product(base_axis_lengths_from_axes),
            ):
                return None
            output_shape = base_axis_lengths_from_axes
            base_axis_groups = [(i,) for i in range(len(base_axes))]

        base_axis_lengths = [
            sympy_product([output_shape[i] for i in group]) if group else sympy.S.One
            for group in base_axis_groups
        ]

        axis_prefix_counts: dict[str, int] = {}
        for axis in base_axes:
            prefix = getattr(axis, "prefix", "")
            axis_prefix_counts[prefix] = axis_prefix_counts.get(prefix, 0) + 1

        def axis_deps_from_index(loc: str, axes: Sequence[Any]) -> Optional[tuple[int, ...]]:
            index = self._input_index(loc)
            if _is_zero_index(index):
                return ()
            free_symbols = index.free_symbols
            deps = [
                i for i, axis in enumerate(axes)
                if axis.symbol() in free_symbols
            ]
            if deps:
                return tuple(deps)
            prefix_deps = tuple(
                i for i, axis in enumerate(axes)
                if any(
                    _is_pointwise_symbol_name(getattr(symbol, "name", str(symbol)))
                    and getattr(symbol, "name", str(symbol))[0] == axis.prefix
                    and axis_prefix_counts.get(axis.prefix, 0) == 1
                    for symbol in free_symbols
                )
            )
            return prefix_deps if prefix_deps else None

        def base_axis_deps_from_shape(
            shape: Sequence[sympy.Expr],
            index_deps: Optional[tuple[int, ...]],
        ) -> Optional[tuple[int, ...]]:
            shape = list(shape)

            if matches(sympy_product(shape), sympy.S.One):
                return ()

            candidates: list[tuple[int, ...]] = []

            def add_candidate(candidate: tuple[int, ...]) -> None:
                if candidate not in candidates:
                    candidates.append(candidate)

            if len(shape) == len(output_shape):
                deps: list[int] = []
                valid = True
                for axis_index, group in enumerate(base_axis_groups):
                    actual_group = [shape[i] for i in group]
                    output_group = [output_shape[i] for i in group]
                    if shape_matches(actual_group, output_group):
                        deps.append(axis_index)
                    elif all(matches(dim, sympy.S.One) for dim in actual_group):
                        continue
                    else:
                        valid = False
                        break
                if valid:
                    add_candidate(tuple(deps))

            axis_count = len(base_axis_groups)
            for mask in range(1, 1 << axis_count):
                subset = tuple(i for i in range(axis_count) if mask & (1 << i))
                physical_dims = [
                    dim
                    for axis_index in subset
                    for dim in base_axis_groups[axis_index]
                ]
                physical_shape = [output_shape[dim] for dim in physical_dims]
                if shape_matches(shape, physical_shape):
                    add_candidate(subset)
                logical_shape = [base_axis_lengths[axis_index] for axis_index in subset]
                if shape_matches(shape, logical_shape):
                    add_candidate(subset)
                if matches(sympy_product(shape), sympy_product(logical_shape)):
                    add_candidate(subset)

            if index_deps is not None:
                for candidate in candidates:
                    if candidate == index_deps:
                        return candidate

            if len(candidates) == 1:
                return candidates[0]
            return None

        base_axis_deps: dict[str, tuple[int, ...]] = {}
        input_shapes: dict[str, Sequence[sympy.Expr]] = {}
        for name, (_, loc, _) in self._input_items():
            try:
                shape = V.graph.get_buffer(name).get_size()
            except Exception:
                return None
            input_shapes[loc] = shape
            index_deps = axis_deps_from_index(loc, base_axes)
            deps = base_axis_deps_from_shape(shape, index_deps)
            if deps is None:
                deps = index_deps
            if deps is None:
                return None
            base_axis_deps[loc] = deps

        def merged_axis_index_groups() -> list[tuple[int, ...]]:
            if any(len(group) != 1 for group in base_axis_groups):
                return [(i,) for i in range(len(base_axis_groups))]

            groups: list[tuple[int, ...]] = []
            current: list[int] = []
            current_signature: Optional[tuple[bool, ...]] = None
            input_locs = [loc for _, (_, loc, _) in self._input_items()]
            for axis_index in range(len(base_axis_groups)):
                signature = tuple(
                    axis_index in base_axis_deps[loc]
                    for loc in input_locs
                )
                if current and signature != current_signature:
                    groups.append(tuple(current))
                    current = []
                current.append(axis_index)
                current_signature = signature
            if current:
                groups.append(tuple(current))
            return groups

        base_group_index_groups = merged_axis_index_groups()
        axis_groups = [
            tuple(dim for base_index in group for dim in base_axis_groups[base_index])
            for group in base_group_index_groups
        ]
        axis_lengths = [
            sympy_product([base_axis_lengths[base_index] for base_index in group])
            for group in base_group_index_groups
        ]
        if len(axis_groups) <= 1:
            return None

        axis_dim_vars = []
        for group, length in zip(base_group_index_groups, axis_lengths):
            if len(group) == 1:
                base_axis = base_axes[group[0]]
                axis_dim_vars.append(self._axis_dim_var(base_axis, length))
            else:
                axis_dim_vars.append(self._static_value_var("B", length))

        axis_names = self._pointwise_logical_axis_names(len(axis_groups))
        full_shape = _tilelang_shape_tuple_expr(axis_dim_vars)
        tensor_shapes: dict[str, TileLangTensor] = {
            var: TileLangTensor.from_shape(
                axis_dim_vars,
                dtype,
                dims=axis_names,
            )
            for var, dtype in output_dtypes.items()
        }

        def logical_deps_from_base_deps(base_deps: tuple[int, ...]) -> Optional[tuple[int, ...]]:
            base_dep_set = set(base_deps)
            deps: list[int] = []
            for axis_index, group in enumerate(base_group_index_groups):
                present = [base_index in base_dep_set for base_index in group]
                if all(present):
                    deps.append(axis_index)
                elif any(present):
                    return None
            return tuple(deps)

        def input_axis_positions(
            shape: Sequence[sympy.Expr],
            deps: tuple[int, ...],
        ) -> Optional[tuple[list[Optional[int]], list[str]]]:
            shape = list(shape)

            if not deps:
                if matches(sympy_product(shape), sympy.S.One):
                    return [None], ["1"]
                return None

            if len(shape) == len(deps):
                for actual, axis_index in zip(shape, deps):
                    if not matches(actual, axis_lengths[axis_index]):
                        return None
                return list(deps), [axis_dim_vars[axis_index] for axis_index in deps]

            if len(shape) == len(axis_groups):
                positions: list[Optional[int]] = []
                dims: list[str] = []
                dep_set = set(deps)
                valid = True
                for axis_index, actual in enumerate(shape):
                    if axis_index in dep_set:
                        if not matches(actual, axis_lengths[axis_index]):
                            valid = False
                            break
                        positions.append(axis_index)
                        dims.append(axis_dim_vars[axis_index])
                    else:
                        if not matches(actual, sympy.S.One):
                            valid = False
                            break
                        positions.append(None)
                        dims.append("1")
                if valid:
                    return positions, dims

            if len(shape) == len(output_shape):
                positions = []
                dims = []
                for axis_index, group in enumerate(axis_groups):
                    actual_group = [shape[i] for i in group]
                    output_group = [output_shape[i] for i in group]
                    if shape_matches(actual_group, output_group):
                        positions.append(axis_index)
                        dims.append(axis_dim_vars[axis_index])
                    elif all(matches(dim, sympy.S.One) for dim in actual_group):
                        positions.append(None)
                        dims.append("1")
                    else:
                        return None
                return positions, dims

            dep_output_dims = [
                dim
                for axis_index in deps
                for dim in axis_groups[axis_index]
            ]
            dep_output_shape = [output_shape[dim] for dim in dep_output_dims]
            if shape_matches(shape, dep_output_shape):
                return list(deps), [axis_dim_vars[axis_index] for axis_index in deps]

            if len(deps) == 1 and matches(sympy_product(shape), axis_lengths[deps[0]]):
                axis_index = deps[0]
                return [axis_index], [axis_dim_vars[axis_index]]

            return None

        input_positions: dict[str, list[Optional[int]]] = {}
        axis_deps: dict[str, tuple[int, ...]] = {}

        def dim_names_for_positions(positions: Sequence[Optional[int]]) -> list[str]:
            if len(positions) == len(axis_names):
                return axis_names
            dims: list[str] = []
            for i, axis_index in enumerate(positions):
                if axis_index is None:
                    dims.append(axis_names[i] if i < len(axis_names) else f"d{i}")
                else:
                    dims.append(axis_names[axis_index])
            return dims

        for name, (var, loc, dtype) in self._input_items():
            shape = input_shapes[loc]
            deps = logical_deps_from_base_deps(base_axis_deps[loc])
            if deps is None:
                return None
            input_layout = input_axis_positions(shape, deps)
            if input_layout is None:
                return None
            positions, dims = input_layout
            tensor_shapes[var] = TileLangTensor.from_shape(
                dims,
                dtype,
                dims=dim_names_for_positions(positions),
            )
            axis_deps[loc] = deps
            input_positions[loc] = positions

        axis_block_vars = tuple(
            f"_{axis_name.upper()}BLOCK" for axis_name in axis_names
        )
        block = TileLangBlock.create(
            "pointwise_ranked",
            dict(zip(axis_names, axis_block_vars)),
        )
        logical_local_tensor = TileLangTensor.from_shape(
            axis_dim_vars,
            torch.float32,
            dims=axis_names,
        )
        self._tl_pointwise_axis_lengths = axis_lengths
        self._tl_pointwise_axis_names = axis_names
        return PointwiseLayoutPlan(
            kind="ranked",
            full_shape=full_shape,
            local_shape=logical_local_tensor.shape_expr(block=block),
            tensor_shapes=tensor_shapes,
            dims=tuple(axis_names),
            block=block,
            axis_lengths=tuple(axis_lengths),
            axis_dim_vars=tuple(axis_dim_vars),
            axis_block_vars=axis_block_vars,
            axis_deps=axis_deps,
            input_positions=input_positions,
        )

    def _pointwise_linear_layout_plan(self) -> PointwiseLayoutPlan:
        full_shape = "(_xnumel,)"
        block = self._linear_pointwise_block()
        local_shape = TileLangTensor.from_shape(
            full_shape,
            torch.float32,
            dims=("x",),
        ).shape_expr(block=block)
        tensor_shapes: dict[str, TileLangTensor] = {
            var: TileLangTensor.from_shape(full_shape, dtype, dims=("x",))
            for _, (var, _, dtype) in self._output_buffer_items()
        }
        for _, (var, loc, dtype) in self._input_items():
            kind = self._pointwise_input_kind(loc)
            if kind == "scalar":
                tensor_shapes[var] = TileLangTensor.from_shape(
                    ("1",),
                    dtype,
                    dims=("x",),
                )
            elif kind == "broadcast":
                tensor_shapes[var] = TileLangTensor.from_shape(
                    self._pointwise_broadcast_shape(loc),
                    dtype,
                    dims=("x",),
                )
            else:
                tensor_shapes[var] = TileLangTensor.from_shape(
                    full_shape,
                    dtype,
                    dims=("x",),
                )
        return PointwiseLayoutPlan(
            kind="linear",
            full_shape=full_shape,
            local_shape=local_shape,
            tensor_shapes=tensor_shapes,
            dims=("x",),
            block=block,
        )

    def _pointwise_layout_plan(self) -> PointwiseLayoutPlan:
        ranked_plan = self._pointwise_ranked_broadcast_info()
        if ranked_plan is not None:
            return ranked_plan
        return self._pointwise_linear_layout_plan()

    def _reduction_input_kind(self, loc: str) -> str:
        index = self._input_index(loc)
        if _is_zero_index(index):
            return "scalar"
        has_x = _has_pointwise_index(index)
        has_r = _has_reduction_index(index)
        if has_x and has_r:
            return "matrix"
        if has_r:
            return "col_vector"
        return "row_broadcast"

    def _reduction_symbol_replacements(
        self,
        index: sympy.Expr,
        reduction_index: str = "_tl_i",
    ) -> dict[sympy.Symbol, str]:
        r_symbols = [
            symbol for symbol in index.free_symbols
            if _is_reduction_symbol_name(getattr(symbol, "name", str(symbol)))
        ]
        if len(r_symbols) <= 1:
            return {}

        def sort_key(symbol: sympy.Symbol):
            coeff = sympy.expand(index).coeff(symbol)
            if coeff is not None and not getattr(coeff, "free_symbols", None):
                try:
                    return int(coeff)
                except Exception:
                    pass
            node = (
                self.range_tree_nodes.get(symbol)
                or self.range_tree_nodes_removed.get(symbol)
            )
            if node is not None:
                return node.sorted_order or 0
            return str(symbol)

        ordered = sorted(r_symbols, key=sort_key)
        replacements: dict[sympy.Symbol, str] = {}
        factor = 1
        for i, symbol in enumerate(ordered):
            node = (
                self.range_tree_nodes.get(symbol)
                or self.range_tree_nodes_removed.get(symbol)
            )
            if node is None:
                raise NotImplementedError(
                    "TileLang reduction could not determine range for "
                    f"index symbol {symbol}"
                )
            try:
                extent = int(node.length)
            except Exception as exc:
                raise NotImplementedError(
                    "TileLang reduction only supports static expanded "
                    f"reduction axes; {symbol} has length {node.length}"
                ) from exc

            if i == 0:
                replacements[symbol] = f"({reduction_index} % {extent})"
            elif i == len(ordered) - 1:
                replacements[symbol] = f"({reduction_index} // {factor})"
            else:
                replacements[symbol] = f"(({reduction_index} // {factor}) % {extent})"
            factor *= extent
        return replacements

    def _axis_extent(self, symbol: sympy.Symbol) -> sympy.Expr:
        node = (
            self.range_tree_nodes.get(symbol)
            or self.range_tree_nodes_removed.get(symbol)
        )
        if node is not None:
            return node.length
        name = getattr(symbol, "name", str(symbol))
        if _is_pointwise_symbol_name(name):
            return self.numels.get(name[0], sympy.S.One)
        if _is_reduction_symbol_name(name):
            return self.numels.get("r", sympy.S.One)
        return sympy.S.One

    def _static_axis_var(self, symbol: sympy.Symbol) -> str:
        if symbol not in self._tl_static_axis_vars:
            name = f"_R{len(self._tl_static_axis_vars)}"
            self._tl_static_axis_vars[symbol] = (name, self._axis_extent(symbol))
        return self._tl_static_axis_vars[symbol][0]

    def _static_value_var(self, prefix: str, value: sympy.Expr) -> str:
        key = (prefix, str(value))
        if key not in self._tl_static_value_vars:
            name = f"_{prefix}{len(self._tl_static_value_vars)}"
            self._tl_static_value_vars[key] = (name, value)
        return self._tl_static_value_vars[key][0]

    def tilelang_static_axis_args(self) -> list[tuple[str, sympy.Expr]]:
        return list(dict.fromkeys([
            *self._tl_static_axis_vars.values(),
            *self._tl_static_value_vars.values(),
        ]))

    def _index_symbols_by_stride(self, index: sympy.Expr) -> list[sympy.Symbol]:
        def axis_order(symbol: sympy.Symbol) -> int:
            node = self.range_tree_nodes.get(symbol)
            if node is None:
                node = self.range_tree_nodes_removed.get(symbol)
            if node is not None and node.sorted_order is not None:
                return int(node.sorted_order)
            return -1

        def stride(symbol: sympy.Symbol) -> tuple[int, int, str]:
            coeff = sympy.expand(index).coeff(symbol)
            try:
                if coeff is not None and not getattr(coeff, "free_symbols", None):
                    stride_hint = int(coeff)
                else:
                    stride_hint = int(V.graph.sizevars.size_hint(coeff))
            except Exception:
                stride_hint = axis_order(symbol)

            # ``index.free_symbols`` is a set.  Always include a deterministic
            # tie-breaker so a symbolic stride such as ``s1`` cannot collide
            # with a concrete stride or another unresolved symbolic stride.
            return (
                stride_hint,
                axis_order(symbol),
                getattr(symbol, "name", str(symbol)),
            )

        symbols = [
            symbol for symbol in index.free_symbols
            if _is_pointwise_symbol_name(getattr(symbol, "name", str(symbol)))
            or _is_reduction_symbol_name(getattr(symbol, "name", str(symbol)))
        ]
        return sorted(symbols, key=stride, reverse=True)

    def _reduction_matrix_symbols(self, index: sympy.Expr) -> list[sympy.Symbol]:
        return self._index_symbols_by_stride(index)

    def _reduction_pointwise_tile_symbol(self, index: sympy.Expr) -> Optional[sympy.Symbol]:
        pointwise_symbols = [
            symbol for symbol in self._reduction_matrix_symbols(index)
            if _is_pointwise_symbol_name(getattr(symbol, "name", str(symbol)))
        ]
        return pointwise_symbols[0] if pointwise_symbols else None

    def _reduction_matrix_dim_expr(self, symbol: sympy.Symbol, index: sympy.Expr) -> str:
        symbols = self._reduction_matrix_symbols(index)
        name = getattr(symbol, "name", str(symbol))
        if name in self._tl_reduction_dim_names:
            return self._static_axis_var(symbol)
        if _is_pointwise_symbol_name(name) and len([
            s for s in symbols
            if _is_pointwise_symbol_name(getattr(s, "name", str(s)))
        ]) == 1:
            return f"_{name[0]}numel"
        if _is_reduction_symbol_name(name) and len([
            s for s in symbols
            if _is_reduction_symbol_name(getattr(s, "name", str(s)))
        ]) == 1:
            return "_rnumel"
        return self._static_axis_var(symbol)

    def _representative_reduction_matrix_index(self) -> Optional[sympy.Expr]:
        for _, (_, loc, _) in self._input_items():
            index = self._input_index(loc)
            if _has_pointwise_index(index) and _has_reduction_index(index):
                return index
        for loc in self._outputs:
            index = self._output_index(loc)
            if _has_pointwise_index(index) and _has_reduction_index(index):
                return index
        return None

    def _pointwise_symbol_replacements(
        self,
        index: sympy.Expr,
        pointwise_index: str = "cid",
    ) -> dict[sympy.Symbol, str]:
        pointwise_symbols = [
            symbol for symbol in index.free_symbols
            if _is_pointwise_symbol_name(getattr(symbol, "name", str(symbol)))
        ]
        if len(pointwise_symbols) <= 1:
            return {symbol: pointwise_index for symbol in pointwise_symbols}

        ordered = self._index_symbols_by_stride(
            sum(symbol * sympy.expand(index).coeff(symbol) for symbol in pointwise_symbols)
        )
        ordered = [
            symbol for symbol in ordered
            if _is_pointwise_symbol_name(getattr(symbol, "name", str(symbol)))
        ]
        replacements: dict[sympy.Symbol, str] = {}
        factor = 1
        for i, symbol in enumerate(reversed(ordered)):
            extent = self._axis_extent(symbol)
            try:
                extent_int = int(extent)
            except Exception as exc:
                raise NotImplementedError(
                    "TileLang reduction only supports static expanded "
                    f"pointwise axes; {symbol} has length {extent}"
                ) from exc
            if i == 0:
                replacements[symbol] = f"({pointwise_index} % {extent_int})"
            elif i == len(ordered) - 1:
                replacements[symbol] = f"({pointwise_index} // {factor})"
            else:
                replacements[symbol] = f"(({pointwise_index} // {factor}) % {extent_int})"
            factor *= extent_int
        return replacements

    def _symbol_dim_name(self, symbol: sympy.Symbol) -> str:
        name = getattr(symbol, "name", str(symbol))
        if name in self._tl_reduction_dim_names:
            return self._tl_reduction_dim_names[name]
        if _is_reduction_symbol_name(name):
            return "r"
        if _is_pointwise_symbol_name(name):
            return name[0]
        return name

    def _reduction_matrix_tensor_from_index(
        self,
        index: sympy.Expr,
        dtype: torch.dtype,
    ) -> TileLangTensor:
        reduction_numel = self.numels.get("r", sympy.S.One)
        if _is_dense_reduction_matrix_index(index, reduction_numel):
            pointwise_symbols = [
                symbol
                for symbol in self._index_symbols_by_stride(index)
                if _is_pointwise_symbol_name(
                    getattr(symbol, "name", str(symbol))
                )
            ]
            if len(pointwise_symbols) == 1:
                pointwise_symbol = pointwise_symbols[0]
                pointwise_name = getattr(
                    pointwise_symbol,
                    "name",
                    str(pointwise_symbol),
                )
                logical_dim = self._tl_reduction_dim_names.get(pointwise_name)
                if logical_dim is not None:
                    return TileLangTensor.from_shape(
                        (
                            self._static_axis_var(pointwise_symbol),
                            "_rnumel",
                        ),
                        dtype,
                        dims=(logical_dim, "r"),
                    )
            return TileLangTensor.from_shape(
                ("_xnumel", "_rnumel"),
                dtype,
                dims=("x", "r"),
            )

        dims: list[str] = []
        current_shape: list[str] = []
        symbols = self._index_symbols_by_stride(index)
        pointwise_tile_symbol = self._reduction_pointwise_tile_symbol(index)
        for symbol in symbols:
            name = getattr(symbol, "name", str(symbol))
            if name in self._tl_reduction_dim_names:
                dims.append(self._tl_reduction_dim_names[name])
            elif (
                _is_pointwise_symbol_name(name)
                and symbol != pointwise_tile_symbol
            ):
                dims.append(name)
            else:
                dims.append(self._symbol_dim_name(symbol))
            current_shape.append(self._reduction_matrix_dim_expr(symbol, index))
        return TileLangTensor.from_shape(current_shape, dtype, dims=dims)

    def _reduction_scalar_tensor_from_index(
        self,
        index: sympy.Expr,
        dtype: torch.dtype,
    ) -> TileLangTensor:
        return self._reduction_matrix_tensor_from_index(
            index,
            dtype,
        ).with_dim_extents({"r": "1"})

    def _reduction_matrix_index_for_output(self, loc: str) -> sympy.Expr:
        index = self._output_index(loc)
        if not (_has_pointwise_index(index) and _has_reduction_index(index)):
            representative_index = self._representative_reduction_matrix_index()
            if representative_index is not None:
                return representative_index
        return index

    def _reduction_matrix_start_index_tuple(
        self,
        index: sympy.Expr,
        pointwise_index: str = "cid * _XBLOCK",
        reduction_index: str = "0",
    ) -> str:
        reduction_numel = self.numels.get("r", sympy.S.One)
        if _is_dense_reduction_matrix_index(index, reduction_numel):
            return f"{pointwise_index}, {reduction_index}"

        replacements = {}
        replacements.update(self._pointwise_symbol_replacements(index, pointwise_index))
        replacements.update(self._reduction_symbol_replacements(index, reduction_index))
        for symbol in index.free_symbols:
            if _is_reduction_symbol_name(getattr(symbol, "name", str(symbol))):
                replacements.setdefault(symbol, reduction_index)
        symbols = self._index_symbols_by_stride(index)
        parts = [
            _tilelang_index_expr(symbol, symbol_replacements=replacements)
            for symbol in symbols
        ]
        return ", ".join(parts)

    def _is_tensor_output_loc(
        self,
        loc: str,
        tensor_arg_names: Optional[set[str]] = None,
    ) -> bool:
        for _, (var, out_loc, _) in self._output_buffer_items():
            if out_loc != loc:
                continue
            return tensor_arg_names is None or var in tensor_arg_names
        return False

    def _valid_vector_epilogue_locs(
        self,
        tensor_arg_names: Optional[set[str]] = None,
    ) -> set[str]:
        return {
            loc
            for loc in self._outputs
            if loc not in self._reduction_outputs
            and self._is_tensor_output_loc(loc, tensor_arg_names)
            and _has_reduction_index(self._output_index(loc))
        }

    def _reduction_output_extent(
        self,
        loc: str,
        tensor_arg_names: Optional[set[str]] = None,
    ) -> str:
        if loc in self._reduction_outputs:
            return "scalar"
        if loc in self._valid_vector_epilogue_locs(tensor_arg_names):
            return "matrix"
        return "scalar"

    def _reduction_input_tensor(
        self,
        loc: str,
        dtype: torch.dtype,
    ) -> TileLangTensor:
        kind = self._reduction_input_kind(loc)
        if kind == "matrix":
            return self._reduction_matrix_tensor_from_index(
                self._input_index(loc),
                dtype,
            )
        if kind == "row_broadcast":
            return TileLangTensor.from_shape(
                ("_xnumel", "1"),
                dtype,
                dims=("x", "r"),
            )
        return TileLangTensor.from_shape(
            ("1", "_rnumel"),
            dtype,
            dims=("x", "r"),
        )

    def _reduction_output_tensor(
        self,
        loc: str,
        dtype: torch.dtype,
        tensor_arg_names: Optional[set[str]] = None,
    ) -> TileLangTensor:
        index = self._reduction_matrix_index_for_output(loc)
        extent = self._reduction_output_extent(loc, tensor_arg_names)
        if _is_zero_index(index):
            index = self._representative_reduction_matrix_index()
        if index is None:
            return TileLangTensor.from_shape(
                ("1", "_rnumel") if extent == "matrix" else ("1", "1"),
                dtype,
                dims=("x", "r"),
            )
        if extent == "matrix":
            if (
                self._representative_reduction_matrix_index() is None
                and _has_reduction_index(index)
                and not _has_pointwise_index(index)
            ):
                return TileLangTensor.from_shape(
                    ("1", "_rnumel"),
                    dtype,
                    dims=("x", "r"),
                )
            return self._reduction_matrix_tensor_from_index(index, dtype)
        return self._reduction_scalar_tensor_from_index(index, dtype)

    def codegen_kernel(self, name: Optional[str] = None) -> str:
        """Build a program plan, then emit it without codegen-time graph queries."""
        if self._reduction_outputs:
            plan = TileLangReductionProgramBuilder(self).build(name)
        else:
            plan = TileLangPointwiseProgramBuilder(self).build(name)
        src = KernelProgramEmitter().emit(plan)
        return src

    def _check_op_graph_dtype(
        self,
        var_name: str,
        dtype: torch.dtype,
        _visited: Optional[set] = None,
    ) -> None:
        """
        Walk the op graph from var_name and raise NotImplementedError if any
        op does not support `dtype`.  Called from store() so it executes
        inside _body(*index_vars) and triggers inductor's Triton fallback.
        """
        if _visited is None:
            _visited = set()
        if var_name in _visited:
            return
        _visited.add(var_name)

        value_info = self._values.get(var_name)
        op_record = value_info.op if value_info is not None else None
        if op_record is None:
            return  # input buffer or constant; no op to check

        op_name = op_record.name
        operands = op_record.operands
        recorded_dtype = op_record.dtype

        if op_name != "to_dtype":
            spec = _tilelang_op(op_name)
            operand_dtypes = [
                (
                    self._values.get(str(operand)).dtype
                    if self._values.get(str(operand)) is not None
                    else None
                )
                for operand in operands
            ]
            execution_dtype = _select_vec_op_execution_dtype(
                spec,
                operand_dtypes,
                value_info.dtype if value_info is not None else None,
                recorded_dtype,
                dtype,
            )
            lowered_spec, _ = spec.lower(operands, execution_dtype)
            lowered_spec.validate_dtype(execution_dtype)
        else:
            execution_dtype = recorded_dtype or dtype

        for op in operands:
            operand_info = self._values.get(str(op))
            self._check_op_graph_dtype(
                str(op),
                (
                    (operand_info.dtype if operand_info is not None else None)
                    or execution_dtype
                ),
                _visited,
            )

    def _tilelang_buffer_dtypes(self) -> dict[str, torch.dtype]:
        buffer_dtypes = {
            loc: buffer_info.dtype
            for loc, buffer_info in self._buffers.items()
        }
        for loc, (_, buffer_dtype) in self._reduction_outputs.items():
            buffer_dtypes[loc] = buffer_dtype
        return buffer_dtypes

    def _pointwise_local_tensors(
        self,
        plan: PointwiseLayoutPlan,
    ) -> dict[str, TileLangTensor]:
        local_tensors: dict[str, TileLangTensor] = {}
        for _, (_, loc, dtype) in [
            *self._input_items(),
            *self._output_buffer_items(),
        ]:
            tensor = TileLangTensor.from_shape(
                plan.full_shape,
                dtype,
                dims=plan.dims,
            )
            local_tensors[loc] = tensor
        return local_tensors

    def call_kernel(
        self, name: str, origin_node: Optional[ir.IRNode] = None
    ) -> None:
        wrapper = V.graph.wrapper_code
        _, call_args, signature, _ = self.args.python_argdefs()
        tensor_args = [a for a, s in zip(call_args, signature) if isinstance(s, TensorArg)]
        numel_args  = [str(tree.numel) for tree in self.active_range_trees()]
        axis_args = [
            str(extent) for _name, extent in self.tilelang_static_axis_args()
        ]
        wrapper.writeline(
            f"{name}({', '.join(tensor_args + numel_args + axis_args)})"
        )

    def create_cse_var(self, name, bounds=None, dtype=None, shape=None) -> TileLangCSEVariable:
        var = TileLangCSEVariable(name, bounds, dtype, shape)
        if dtype is not None:
            self._record_value_dtype(name, dtype)
        # Consume any pending op set by TileLangOverrides.*
        # _pending_op is cleared in load() before load-expr generate calls,
        # so only compute-expression vars pick it up here.
        pending_op = self._consume_pending_compute_op()
        if pending_op is not None:
            op_name = pending_op.name
            operands = pending_op.operands
            op_dtype = pending_op.dtype
            if dtype is None and op_dtype is not None:
                var.dtype = op_dtype
                self._record_value_dtype(name, op_dtype)
            if op_name == "const":
                self._record_value_const(name, operands[0])
            else:
                self._record_value_op(name, pending_op)
        return var

    def should_use_persistent_reduction(self) -> bool:
        # Keep the same semantic contract as Triton: persistent reductions
        # cover the complete reduction domain in one tile, while
        # non-persistent reductions require the TileLang RBLOCK loop.
        if not self.inside_reduction or not config.triton.persistent_reductions:
            return False

        threshold = {
            ReductionHint.INNER: 4096,
            ReductionHint.DEFAULT: 4096,
        }.get(self.features.get_reduction_hint(), 64)
        if self.cooperative_reduction:
            numel = self.features.numel
            if isinstance(numel, (list, tuple)):
                numel = sympy_product(numel)
            try:
                threshold *= 32 // min(V.graph.sizevars.size_hint(numel), 32)
            except ValueError:
                pass  # unbacked symint

        if config.triton.multi_kernel:
            threshold *= 16

        reduction_numel = self.features.reduction_numel
        if isinstance(reduction_numel, (list, tuple)):
            reduction_numel = sympy_product(reduction_numel)
        return V.graph.sizevars.statically_known_leq(
            reduction_numel,
            threshold,
        )

    def should_use_cooperative_reduction(self) -> bool:
        return False


@dataclasses.dataclass
class TileLangPointwiseProgramBuilder:
    """Lowers captured pointwise state into a typed kernel program plan."""

    kernel: TileLangKernel

    @staticmethod
    def _broadcast_dims(
        source: TileLangTensor,
        destination: TileLangTensor,
    ) -> tuple[str, ...]:
        return tuple(
            dim
            for dim, source_extent, destination_extent in zip(
                source.dims,
                source.current_shape,
                destination.current_shape,
            )
            if source_extent != destination_extent
        )

    def _signature(
        self,
        name: Optional[str],
        layout: PointwiseLayoutPlan,
    ) -> KernelSignaturePlan:
        prim_fn_name = f"{name or str(Placeholder.KERNEL_NAME)}_prim_fn"
        argdefs, _, signature, _ = self.kernel.args.python_argdefs()
        parameters: list[str] = []
        for argdef, sig in zip(argdefs, signature):
            if not isinstance(sig, TensorArg):
                continue
            tensor = layout.tensor_shapes.get(argdef.name)
            if tensor is None:
                tensor = TileLangTensor.from_shape(
                    layout.full_shape,
                    sig.dtype,
                    dims=layout.dims or None,
                )
            parameters.append(f"{argdef.name}: {tensor.tensor_type()}")
        return KernelSignaturePlan(prim_fn_name, tuple(parameters))

    def _plan_vec_compute(
        self,
        layout: PointwiseLayoutPlan,
        protected_buffers: set[str],
        buffer_shapes: dict[str, str],
        buffer_dtypes: dict[str, torch.dtype],
    ) -> tuple[
        Optional[PointwiseOpPlan],
        tuple[BufferPlan, ...],
        tuple[TransferPlan, ...],
    ]:
        raw_operations: list[VecOp] = []
        computed_outputs: list[str] = []
        passthroughs: list[TransferPlan] = []
        default_dtype: Optional[torch.dtype] = None

        for out_loc, (result_var, dtype) in self.kernel._output_items():
            default_dtype = default_dtype or dtype
            output_operations: list[VecOp] = []
            source = _build_vec_ops(
                result_var,
                out_loc,
                output_operations,
                self.kernel._values,
            )
            if not output_operations:
                if str(source) != out_loc:
                    if str(source) not in buffer_shapes:
                        raise TileLangUnsupported(
                            "pointwise capture", f"unrecorded output expression: {source}"
                        )
                    passthroughs.append(
                        TransferPlan("COPY", str(source), out_loc)
                    )
                continue

            _finalize_vec_ops_output(output_operations, out_loc)
            raw_operations.extend(
                _normalize_vec_op_dtypes(
                    output_operations,
                    buffer_dtypes,
                    dtype,
                )
            )
            computed_outputs.append(out_loc)

        if not raw_operations:
            return None, (), tuple(passthroughs)
        if default_dtype is None:
            raise RuntimeError("pointwise vector plan has no output dtype")

        vec_region, temporary_allocations = KernelBufferPlanner().plan_vec_region(
            raw_operations,
            local_shape=layout.local_shape,
            default_dtype=default_dtype,
            protected_buffers=protected_buffers,
            buffer_shapes=buffer_shapes,
            buffer_dtypes=buffer_dtypes,
        )
        input_locs = tuple(
            loc for _, (_, loc, _) in self.kernel._input_items()
        )
        return (
            PointwiseOpPlan(
                op_id="pointwise_0",
                inputs=input_locs,
                outputs=tuple(computed_outputs),
                origin=OperationOrigin("pointwise", "captured_outputs"),
                compute=vec_region,
            ),
            temporary_allocations,
            tuple(passthroughs),
        )

    def build(self, name: Optional[str] = None) -> TileLangKernelPlan:
        kernel = self.kernel
        layout = kernel._pointwise_layout_plan()
        local_tensors = kernel._pointwise_local_tensors(layout)
        allocations: list[BufferPlan] = []
        buffer_shapes: dict[str, str] = {}
        buffer_dtypes = kernel._tilelang_buffer_dtypes()
        allocated: set[str] = set()
        input_transfers: list[TransferPlan] = []
        output_transfers: list[TransferPlan] = []
        broadcast_requests: list[BroadcastRequest] = []
        external_values: set[str] = set()

        def add_allocation(
            name: str,
            tensor: TileLangTensor,
            *,
            shape: Optional[str] = None,
        ) -> None:
            if name in allocated:
                return
            materialized_shape = shape or tensor.shape_expr(block=layout.block)
            allocation = BufferPlan(
                name=name,
                shape=materialized_shape,
                dtype=tensor.dtype,
            )
            allocations.append(allocation)
            buffer_shapes[name] = materialized_shape
            buffer_dtypes[name] = tensor.dtype
            allocated.add(name)

        if layout.kind == "linear":
            loop = kernel._tilelang_for_loop("X0BLOCK")
            loops = {"x": loop}
            for _, (var, loc, dtype) in kernel._input_items():
                destination_tensor = local_tensors[loc]
                kind = kernel._pointwise_input_kind(loc)
                if kind == "scalar":
                    source_tensor = TileLangTensor.from_shape(
                        ("1",),
                        dtype,
                        dims=("x",),
                    )
                    add_allocation(loc, source_tensor, shape="(1,)")
                    input_transfers.append(
                        TransferPlan(
                            "COPY",
                            source_tensor.tiled_ref(
                                var,
                                loops,
                                local=False,
                                singleton_as_slice=False,
                            ),
                            loc,
                        )
                    )
                    # Broadcast by indexing lane zero inside the consumer VF.
                    external_values.add(loc)
                elif kind == "broadcast":
                    add_allocation(loc, destination_tensor)
                    input_transfers.append(
                        TransferPlan(
                            "INDEXED_LOAD",
                            var,
                            loc,
                            extent="_remain_X",
                            source_index=kernel._pointwise_broadcast_lane_index(
                                loc,
                                pointwise_index="(_tl_base + _tl_i)",
                            ),
                        )
                    )
                    external_values.add(loc)
                else:
                    add_allocation(loc, destination_tensor)
                    input_transfers.append(
                        TransferPlan(
                            "COPY",
                            destination_tensor.tiled_ref(
                                var,
                                loops,
                                local=False,
                            ),
                            destination_tensor.tiled_ref(
                                loc,
                                loops,
                                local=True,
                            ),
                        )
                    )
                    external_values.add(loc)

            region_loops = (loop,)
            serial_loop: Optional[TileLangForLoop] = loop
            region_prologue = (
                "_tl_x_base = cid * _XBLOCK",
                "_tl_outer_remain = T.min(_XBLOCK, _xnumel - _tl_x_base)",
            )
            body_prologue = (
                "_tl_base = cid * _XBLOCK + _tl_block * _XBLOCK_SUB",
                "_remain_X = T.min(_XBLOCK_SUB, T.min("
                "_XBLOCK - _tl_block * _XBLOCK_SUB, _xnumel - _tl_base))",
            )
            launch = KernelLaunchPlan("T.ceildiv(_xnumel, _XBLOCK)")
            preamble = (
                "_XBLOCK = X0BLOCK",
                "_XBLOCK_SUB = X0BLOCK_SUB",
            )
            loop_map = loops
        elif layout.kind == "ranked":
            axis_dim_vars = list(layout.axis_dim_vars)
            axis_names = list(layout.dims)
            axis_deps = layout.axis_deps
            input_positions = layout.input_positions
            axis_block_vars = list(layout.axis_block_vars)
            rank = len(axis_names)
            if len(axis_block_vars) != rank:
                raise RuntimeError(
                    "ranked pointwise plan requires one block per logical axis"
                )
            axis_loops = tuple(
                TileLangForLoop(
                    name=f"pointwise_axis{axis_index}",
                    extent=axis_dim_vars[axis_index],
                    block=axis_block_vars[axis_index],
                    loop_var=f"_tl_axis{axis_index}_pid",
                    base=f"_tl_axis{axis_index}_base",
                    remain=f"_tl_remain_axis{axis_index}",
                )
                for axis_index, axis_name in enumerate(axis_names)
            )
            loops = {
                axis_name: axis_loop
                for axis_name, axis_loop in zip(axis_names, axis_loops)
            }

            def ranked_tensor(
                shape: Sequence[str],
                dtype: torch.dtype,
                dims: Optional[Sequence[str]] = None,
            ) -> TileLangTensor:
                return TileLangTensor.from_shape(
                    shape,
                    dtype,
                    dims=dims or axis_names,
                )

            def input_ref(
                var: str,
                positions: Sequence[Optional[int]],
                dtype: torch.dtype,
            ) -> str:
                if not positions:
                    return _tilelang_ref(var, ("0",))
                if all(position is None for position in positions):
                    if len(positions) == 1:
                        return _tilelang_ref(var, ("0",))
                shape: list[str] = []
                dims: list[str] = []
                for index, axis_index in enumerate(positions):
                    if axis_index is None:
                        shape.append("1")
                        dims.append(axis_names[index] if index < rank else f"d{index}")
                    else:
                        shape.append(axis_dim_vars[axis_index])
                        dims.append(axis_names[axis_index])
                return ranked_tensor(shape, dtype, dims=dims).tiled_ref(
                    var,
                    loops,
                    local=False,
                )

            def broadcast_tensor(
                dependencies: tuple[int, ...],
                dtype: torch.dtype,
            ) -> TileLangTensor:
                dependency_set = set(dependencies)
                return ranked_tensor(
                    [
                        axis_dim_vars[axis_index]
                        if axis_index in dependency_set else "1"
                        for axis_index in range(rank)
                    ],
                    dtype,
                )

            def local_extent_is_one(axis_index: int) -> bool:
                try:
                    return V.graph.sizevars.statically_known_equals(
                        layout.axis_lengths[axis_index],
                        sympy.S.One,
                    )
                except Exception:
                    return False

            def needs_broadcast(dependencies: tuple[int, ...]) -> bool:
                dependency_set = set(dependencies)
                return any(
                    not local_extent_is_one(axis_index)
                    for axis_index in range(rank)
                    if axis_index not in dependency_set
                )

            for _, (var, loc, dtype) in kernel._input_items():
                dependencies = axis_deps[loc]
                positions = input_positions[loc]
                destination_tensor = local_tensors[loc]
                add_allocation(loc, destination_tensor)
                if len(dependencies) == rank:
                    tensor = layout.tensor_shapes[var]
                    input_transfers.append(
                        TransferPlan(
                            "COPY",
                            tensor.tiled_ref(var, loops, local=False),
                            tensor.tiled_ref(loc, loops, local=True),
                        )
                    )
                    external_values.add(loc)
                elif not needs_broadcast(dependencies):
                    tensor = broadcast_tensor(dependencies, dtype)
                    input_transfers.append(
                        TransferPlan(
                            "COPY",
                            input_ref(var, positions, dtype),
                            tensor.tiled_ref(loc, loops, local=True),
                        )
                    )
                    external_values.add(loc)
                else:
                    source_name = f"{loc}_brc_src"
                    source_tensor = broadcast_tensor(dependencies, dtype)
                    add_allocation(source_name, source_tensor)
                    input_transfers.append(
                        TransferPlan(
                            "COPY",
                            input_ref(var, positions, dtype),
                            source_tensor.tiled_ref(
                                source_name,
                                loops,
                                local=True,
                            ),
                        )
                    )
                    missing_axes = [
                        axis_index
                        for axis_index in range(rank)
                        if axis_index not in dependencies
                        and not local_extent_is_one(axis_index)
                    ]
                    current_name = source_name
                    current_tensor = source_tensor
                    for missing_index, axis_index in enumerate(missing_axes):
                        is_last = missing_index == len(missing_axes) - 1
                        next_name = (
                            loc
                            if is_last
                            else f"{loc}_brc_axis{axis_index}"
                        )
                        if is_last:
                            next_tensor = destination_tensor
                        else:
                            next_shape = list(current_tensor.current_shape)
                            next_shape[axis_index] = axis_dim_vars[axis_index]
                            next_tensor = ranked_tensor(
                                next_shape,
                                dtype,
                            )
                            add_allocation(
                                next_name,
                                next_tensor,
                            )
                        broadcast_requests.append(
                            BroadcastRequest(
                                source=current_name,
                                destination=next_name,
                                source_tensor=current_tensor,
                                destination_tensor=next_tensor,
                                broadcast_dims=(axis_names[axis_index],),
                                origin=OperationOrigin(
                                    "input",
                                    var,
                                    "ranked broadcast",
                                ),
                            )
                        )
                        current_name = next_name
                        current_tensor = next_tensor
                    external_values.add(source_name)

            axis_grid_vars = [
                f"_tl_axis{axis_index}_grid" for axis_index in range(rank)
            ]
            region_prologue_list = [
                f"{grid_var} = T.ceildiv({axis_dim_var}, {axis_block_var})"
                for grid_var, axis_dim_var, axis_block_var in zip(
                    axis_grid_vars,
                    axis_dim_vars,
                    axis_block_vars,
                )
            ]
            for axis_index, axis_loop in enumerate(axis_loops):
                suffix_grids = axis_grid_vars[axis_index + 1:]
                suffix_product = " * ".join(suffix_grids)
                if suffix_product:
                    pid_expr = f"((cid // ({suffix_product})) % {axis_grid_vars[axis_index]})"
                else:
                    pid_expr = f"(cid % {axis_grid_vars[axis_index]})"
                region_prologue_list.extend((
                    f"{axis_loop.loop_var} = {pid_expr}",
                    f"{axis_loop.base} = {axis_loop.loop_var} * {axis_loop.block}",
                    f"{axis_loop.remain} = T.min({axis_loop.block}, "
                    f"{axis_loop.extent} - {axis_loop.base})",
                ))

            region_loops = axis_loops
            serial_loop = None
            region_prologue = tuple(region_prologue_list)
            body_prologue = ()
            launch = KernelLaunchPlan(
                " * ".join(
                    f"T.ceildiv({axis_dim_var}, {axis_block_var})"
                    for axis_dim_var, axis_block_var in zip(
                        axis_dim_vars,
                        axis_block_vars,
                    )
                )
            )
            preamble = tuple(
                f"{axis_block_var} = {axis_name.upper()}BLOCK"
                for axis_name, axis_block_var in zip(
                    axis_names,
                    axis_block_vars,
                )
            )
            loop_map = loops
        else:
            raise TileLangUnsupported(
                "pointwise-planning",
                f"unknown pointwise layout kind '{layout.kind}'",
            )

        input_locs = {loc for _, (_, loc, _) in kernel._input_items()}
        for _, (var, loc, _) in kernel._output_buffer_items():
            tensor = local_tensors[loc]
            if loc not in input_locs:
                add_allocation(loc, tensor)
            output_transfers.append(
                TransferPlan(
                    "COPY",
                    tensor.tiled_ref(loc, loop_map, local=True),
                    tensor.tiled_ref(var, loop_map, local=False),
                )
            )

        pointwise_operation, temp_allocations, passthroughs = self._plan_vec_compute(
            layout,
            allocated,
            buffer_shapes,
            buffer_dtypes,
        )
        for allocation in temp_allocations:
            allocations.append(allocation)

        operations: list[KernelOperationPlan] = []
        if pointwise_operation is not None:
            operations.append(pointwise_operation)
        broadcast_requests = [
            dataclasses.replace(
                request,
                consumer_id=(
                    pointwise_operation.op_id
                    if pointwise_operation is not None else None
                ),
            )
            for request in broadcast_requests
        ]
        graph = BroadcastInsertionPass().run(
            operations,
            external_values,
            broadcast_requests,
        )
        region = XTileRegionPlan(
            region_id="x_tile_0",
            loops=region_loops,
            block=layout.block,
            allocations=tuple(allocations),
            input_transfers=tuple(input_transfers),
            operations=(),
            output_transfers=tuple([*passthroughs, *output_transfers]),
            prologue=region_prologue,
            body_prologue=body_prologue,
            serial_loop=serial_loop,
        )
        regions = KernelRegionScheduler.schedule_pointwise(graph, region)
        return TileLangKernelPlan(
            signature=self._signature(name, layout),
            launch=launch,
            operations=graph.operations,
            regions=regions,
            preamble=preamble,
        )


@dataclasses.dataclass
class TileLangReductionProgramBuilder:
    """Lowers captured reductions into an ordered, multi-region kernel plan."""

    kernel: TileLangKernel

    def __post_init__(self) -> None:
        self.r_loop = self.kernel._tilelang_for_loop("RBLOCK")
        self.matrix_index = self.kernel._representative_reduction_matrix_index()
        reduction_axes = [
            axis
            for axis in list(getattr(self.kernel, "sorted_axis", []) or [])
            if getattr(axis, "prefix", "").startswith("r")
        ]
        pointwise_axes = [
            axis
            for axis in list(getattr(self.kernel, "sorted_axis", []) or [])
            if not getattr(axis, "prefix", "").startswith("r")
        ]
        self.ranked = (
            self.matrix_index is not None
            and len(pointwise_axes) > 1
            and len(reduction_axes) == 1
        )
        if self.ranked:
            self.reduction_axis_name = reduction_axes[0].name
            self.pointwise_axis_names = tuple(axis.name for axis in pointwise_axes)
            self.pointwise_axis_lengths = tuple(axis.length for axis in pointwise_axes)
            self.pointwise_axis_dim_vars = tuple(
                self.kernel._static_axis_var(axis.symbol())
                for axis in pointwise_axes
            )
            self.pointwise_axis_block_vars = tuple(
                f"_{axis_name.upper()}BLOCK"
                for axis_name in self.pointwise_axis_names
            )
            self.kernel._tl_reduction_dim_names = {
                getattr(axis.symbol(), "name", str(axis.symbol())): axis.name
                for axis in pointwise_axes
            }
            self.pointwise_loops = tuple(
                TileLangForLoop(
                    name=f"reduction_axis{axis_index}",
                    extent=axis_dim_var,
                    block=axis_block_var,
                    loop_var=f"_tl_axis{axis_index}_pid",
                    base=f"_tl_axis{axis_index}_base",
                    remain=f"_tl_remain_axis{axis_index}",
                )
                for axis_index, (axis_dim_var, axis_block_var) in enumerate(
                    zip(
                        self.pointwise_axis_dim_vars,
                        self.pointwise_axis_block_vars,
                    )
                )
            )
            pointwise_blocks = dict(zip(
                self.pointwise_axis_names,
                self.pointwise_axis_block_vars,
            ))
            self.block = TileLangBlock.create(
                "reduction_ranked",
                {
                    **pointwise_blocks,
                    "r": "_RBLOCK",
                },
            )
            self.loops = {
                **dict(zip(
                    self.pointwise_axis_names,
                    self.pointwise_loops,
                )),
                "r": self.r_loop,
            }
            self.x_loop = None
        else:
            self.block = self.kernel._reduction_tile_block()
            self.x_loop = self.kernel._tilelang_for_loop("X0BLOCK")
            self.loops = {"x": self.x_loop, "r": self.r_loop}
            self.pointwise_axis_names = ()
            self.pointwise_axis_lengths = ()
            self.pointwise_axis_dim_vars = ()
            self.pointwise_axis_block_vars = ()
            self.pointwise_loops = ()
            self.reduction_axis_name = "r0"
        self.operations: list[KernelOperationPlan] = []
        self.broadcast_requests: list[BroadcastRequest] = []
        self.operation_regions: dict[str, str] = {}
        self.broadcast_regions: dict[str, str] = {}
        self.region_drafts: list[KernelRegionPlan] = []
        self.allocations: list[BufferPlan] = []
        self.buffer_tensors: dict[str, TileLangTensor] = {}
        self.buffer_shapes: dict[str, str] = {}
        self.buffer_dtypes = self.kernel._tilelang_buffer_dtypes()
        self.allocated: set[str] = set()
        self.external_values: set[str] = set()
        self.outer_input_transfers: list[TransferPlan] = []
        self.outer_output_transfers: list[TransferPlan] = []
        self.pointwise_id = 0
        self.reduction_id = 0
        self.broadcast_id = 0
        self.region_id = 0

    @staticmethod
    def _domain_key(tensor: TileLangTensor) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return tensor.dims, tensor.current_shape

    @staticmethod
    def _is_singleton(extent: str) -> bool:
        return TileLangBlock._is_broadcast_extent(extent)

    def _join_tensors(
        self,
        tensors: Sequence[TileLangTensor],
        dtype: torch.dtype,
        expected: Optional[TileLangTensor] = None,
    ) -> TileLangTensor:
        candidates = list(tensors)
        if expected is not None:
            candidates.append(expected)
        if not candidates:
            if expected is None:
                raise TileLangUnsupported(
                    "shape-inference",
                    "pointwise result has no tensor domain",
                )
            return dataclasses.replace(expected, dtype=dtype)

        dims = candidates[0].dims
        if any(candidate.dims != dims for candidate in candidates[1:]):
            raise TileLangUnsupported(
                "shape-inference",
                "pointwise operands must use aligned logical dimensions: "
                + ", ".join(
                    f"dims={candidate.dims}, shape={candidate.current_shape}"
                    for candidate in candidates
                ),
            )

        shape: list[str] = []
        for dim_index, dim in enumerate(dims):
            extents = [candidate.current_shape[dim_index] for candidate in candidates]
            non_singletons = {
                extent for extent in extents if not self._is_singleton(extent)
            }
            if len(non_singletons) > 1:
                raise TileLangUnsupported(
                    "shape-inference",
                    f"dimension '{dim}' has incompatible extents {sorted(non_singletons)}",
                )
            shape.append(next(iter(non_singletons), "1"))

        result = TileLangTensor(dims, tuple(shape), dtype)
        if expected is not None and self._domain_key(result) != self._domain_key(expected):
            raise TileLangUnsupported(
                "shape-inference",
                "pointwise result cannot be stored in the declared output domain",
            )
        return result

    def _add_allocation(
        self,
        name: str,
        tensor: TileLangTensor,
        *,
        shape: Optional[str] = None,
    ) -> None:
        materialized_shape = shape or tensor.shape_expr(block=self.block)
        if name in self.allocated:
            if (
                self.buffer_shapes[name] != materialized_shape
                or self.buffer_dtypes[name] != tensor.dtype
            ):
                raise TileLangUnsupported(
                    "buffer-planning",
                    f"buffer '{name}' was requested with incompatible storage",
                )
            return
        allocation = BufferPlan(
            name=name,
            shape=materialized_shape,
            dtype=tensor.dtype,
        )
        self.allocations.append(allocation)
        self.buffer_tensors[name] = tensor
        self.buffer_shapes[name] = materialized_shape
        self.buffer_dtypes[name] = tensor.dtype
        self.allocated.add(name)

    def _input_tensor(self, loc: str, dtype: torch.dtype) -> TileLangTensor:
        kind = self.kernel._reduction_input_kind(loc)
        if kind == "scalar":
            dims = ("x", "r")
            if self.matrix_index is not None:
                dims = self.kernel._reduction_matrix_tensor_from_index(
                    self.matrix_index,
                    dtype,
                ).dims
            return TileLangTensor.from_shape(
                ("1",) * len(dims),
                dtype,
                dims=dims,
            )

        tensor = self.kernel._reduction_input_tensor(loc, dtype)
        if self.matrix_index is None:
            return tensor

        matrix_dims = self.kernel._reduction_matrix_tensor_from_index(
            self.matrix_index,
            dtype,
        ).dims
        if tensor.dims == matrix_dims:
            return tensor
        if not set(tensor.dims).issubset(matrix_dims):
            return tensor
        dim_extents = dict(zip(tensor.dims, tensor.current_shape))
        return TileLangTensor(
            matrix_dims,
            tuple(dim_extents.get(dim, "1") for dim in matrix_dims),
            dtype,
        )

    def _accumulator_tensor(self, dtype: torch.dtype) -> TileLangTensor:
        if self.matrix_index is None:
            return TileLangTensor.from_shape(
                ("1", "1"),
                dtype,
                dims=("x", "r"),
            )
        return self.kernel._reduction_scalar_tensor_from_index(
            self.matrix_index,
            dtype,
        )

    def _register_inputs(self) -> None:
        for _, (var, loc, dtype) in self.kernel._input_items():
            tensor = self._input_tensor(loc, dtype)
            self._add_allocation(loc, tensor)
            self.external_values.add(loc)
            kind = self.kernel._reduction_input_kind(loc)
            if kind == "scalar":
                self.outer_input_transfers.append(
                    TransferPlan("COPY", _tilelang_ref(var, ("0",)), loc)
                )
            elif kind == "row_broadcast":
                self.outer_input_transfers.append(
                    TransferPlan(
                        "COPY",
                        tensor.tiled_ref(var, self.loops, local=False),
                        tensor.tiled_ref(loc, self.loops, local=True),
                    )
                )

    def _register_outputs(self, tensor_arg_names: set[str]) -> None:
        for loc, (_, dtype) in self.kernel._reduction_outputs.items():
            self._add_allocation(loc, self._accumulator_tensor(dtype))
        for _, (var, loc, dtype) in self.kernel._output_buffer_items():
            if var not in tensor_arg_names:
                continue
            tensor = self.kernel._reduction_output_tensor(
                loc,
                dtype,
                tensor_arg_names,
            )
            self._add_allocation(loc, tensor)

    def _signature(
        self,
        name: Optional[str],
        tensor_arg_names: set[str],
    ) -> KernelSignaturePlan:
        argdefs, _, signature, _ = self.kernel.args.python_argdefs()
        input_vars = {var for _, (var, _, _) in self.kernel._input_items()}
        output_vars = {
            var
            for _, (var, _, _) in self.kernel._output_buffer_items()
            if var in tensor_arg_names
        }
        parameters: list[str] = []
        for argdef, sig in zip(argdefs, signature):
            if not isinstance(sig, TensorArg):
                continue
            if argdef.name in input_vars:
                loc = next(
                    loc
                    for _, (var, loc, _) in self.kernel._input_items()
                    if var == argdef.name
                )
                kind = self.kernel._reduction_input_kind(loc)
                if kind == "scalar":
                    tensor = TileLangTensor.from_shape(
                        ("1",),
                        sig.dtype,
                        dims=("x",),
                    )
                else:
                    tensor = self.buffer_tensors[loc]
            elif argdef.name in output_vars:
                loc = next(
                    loc
                    for _, (var, loc, _) in self.kernel._output_buffer_items()
                    if var == argdef.name
                )
                tensor = self.buffer_tensors[loc]
            else:
                tensor = self._accumulator_tensor(sig.dtype)
            parameters.append(f"{argdef.name}: {tensor.tensor_type()}")
        return KernelSignaturePlan(
            f"{name or str(Placeholder.KERNEL_NAME)}_prim_fn",
            tuple(parameters),
        )

    def _collect_input_locs(
        self,
        value: Any,
        visited: Optional[set[str]] = None,
    ) -> set[str]:
        if visited is None:
            visited = set()
        value_name = str(value)
        if value_name in visited:
            return set()
        visited.add(value_name)
        value_info = self.kernel._values.get(value_name)
        if value_info is not None and value_info.buffer is not None:
            buffer_info = self.kernel._buffers.get(value_info.buffer)
            return {value_info.buffer} if buffer_info is not None else set()
        if value_info is None or value_info.op is None:
            return set()
        inputs: set[str] = set()
        for operand in value_info.op.operands:
            inputs.update(self._collect_input_locs(operand, visited))
        return inputs

    def _matrix_load_transfer(self, var: str, loc: str) -> TransferPlan:
        index = self.kernel._input_index(loc)
        tensor = self.buffer_tensors[loc]
        if self.ranked:
            return TransferPlan(
                "COPY",
                tensor.tiled_ref(var, self.loops, local=False),
                tensor.tiled_ref(loc, self.loops, local=True),
            )
        reduction_numel = self.kernel.numels.get("r", sympy.S.One)
        if _is_dense_reduction_matrix_index(index, reduction_numel):
            return TransferPlan(
                "COPY",
                tensor.tiled_ref(var, self.loops, local=False),
                tensor.tiled_ref(loc, self.loops, local=True),
            )
        return TransferPlan(
            "COPY",
            tensor.ref(
                var,
                _split_tilelang_index_tuple(
                    self.kernel._reduction_matrix_start_index_tuple(
                        index,
                        self.x_loop.base,
                        self.r_loop.base,
                    )
                ),
            ),
            loc,
        )

    def _input_transfers(self, required_locs: set[str]) -> tuple[TransferPlan, ...]:
        transfers: list[TransferPlan] = []
        for _, (var, loc, _) in self.kernel._input_items():
            if loc not in required_locs:
                continue
            kind = self.kernel._reduction_input_kind(loc)
            tensor = self.buffer_tensors[loc]
            if kind == "col_vector":
                transfers.append(
                    TransferPlan(
                        "COPY",
                        tensor.tiled_ref(
                            var,
                            self.loops,
                            local=False,
                            singleton_index_dims={"x"},
                        ),
                        tensor.tiled_ref(
                            loc,
                            self.loops,
                            local=True,
                            singleton_index_dims={"x"},
                        ),
                    )
                )
            elif kind == "matrix":
                transfers.append(self._matrix_load_transfer(var, loc))
        return tuple(transfers)

    def _matrix_store_transfer(self, var: str, loc: str) -> TransferPlan:
        index = self.kernel._reduction_matrix_index_for_output(loc)
        tensor = self.buffer_tensors[loc]
        if self.ranked:
            return TransferPlan(
                "COPY",
                tensor.tiled_ref(loc, self.loops, local=True),
                tensor.tiled_ref(var, self.loops, local=False),
            )
        if (
            self.kernel._representative_reduction_matrix_index() is None
            and _has_reduction_index(index)
            and not _has_pointwise_index(index)
        ):
            return TransferPlan(
                "COPY",
                loc,
                tensor.ref(var, {"x": "0", "r": "0"}),
            )
        reduction_numel = self.kernel.numels.get("r", sympy.S.One)
        if _is_dense_reduction_matrix_index(index, reduction_numel):
            return TransferPlan(
                "COPY",
                tensor.tiled_ref(loc, self.loops, local=True),
                tensor.tiled_ref(var, self.loops, local=False),
            )
        return TransferPlan(
            "COPY",
            loc,
            tensor.ref(
                var,
                _split_tilelang_index_tuple(
                    self.kernel._reduction_matrix_start_index_tuple(
                        index,
                        self.x_loop.base,
                        self.r_loop.base,
                    )
                ),
            ),
        )

    def _raw_vec_ops(
        self,
        value: Any,
        target: str,
        dtype: torch.dtype,
    ) -> tuple[Any, list[VecOp]]:
        operations: list[VecOp] = []
        source = _build_vec_ops(
            value,
            target,
            operations,
            self.kernel._values,
        )
        if operations:
            _finalize_vec_ops_output(operations, target)
            operations = _normalize_vec_op_dtypes(
                operations,
                self.buffer_dtypes,
                dtype,
            )
            source = target
        return source, operations

    def _infer_vec_tensors(
        self,
        operations: Sequence[VecOp],
    ) -> dict[str, TileLangTensor]:
        tensors = dict(self.buffer_tensors)
        for operation in operations:
            operand_tensors = [
                tensors[operand]
                for operand in operation.operands
                if isinstance(operand, str) and operand in tensors
            ]
            expected = self.buffer_tensors.get(operation.out)
            dtype = operation.dtype
            if dtype is None:
                raise RuntimeError("VecOp dtype must be normalized before lowering")
            tensors[operation.out] = self._join_tensors(
                operand_tensors,
                dtype,
                expected,
            )
        return tensors

    def _request_broadcast(
        self,
        source: str,
        source_tensor: TileLangTensor,
        destination_tensor: TileLangTensor,
        consumer_id: str,
        region_id: str,
        cache: dict[tuple[Any, ...], str],
    ) -> str:
        destination_tensor = dataclasses.replace(
            destination_tensor,
            dtype=source_tensor.dtype,
        )
        key = (
            source,
            self._domain_key(source_tensor),
            self._domain_key(destination_tensor),
            source_tensor.dtype,
        )
        if key in cache:
            return cache[key]
        destination = f"{source}_brc_{self.broadcast_id}"
        self.broadcast_id += 1
        cache[key] = destination
        self._add_allocation(destination, destination_tensor)
        broadcast_dims = tuple(
            dim
            for dim, source_extent, destination_extent in zip(
                source_tensor.dims,
                source_tensor.current_shape,
                destination_tensor.current_shape,
            )
            if source_extent != destination_extent
        )
        self.broadcast_requests.append(
            BroadcastRequest(
                source=source,
                destination=destination,
                source_tensor=source_tensor,
                destination_tensor=destination_tensor,
                broadcast_dims=broadcast_dims,
                origin=OperationOrigin("broadcast", source, region_id),
                consumer_id=consumer_id,
            )
        )
        self.broadcast_regions[destination] = region_id
        return destination

    def _lower_vec_operations(
        self,
        operations: Sequence[VecOp],
        region_id: str,
        root_target: str,
    ) -> tuple[str, ...]:
        if not operations:
            return ()
        renamed_outputs: dict[str, str] = {}
        namespaced_operations: list[VecOp] = []
        for operation in operations:
            output = operation.out
            if output != root_target:
                output = f"{output}_{region_id}"
            operands = tuple(
                renamed_outputs.get(operand, operand)
                if isinstance(operand, str) else operand
                for operand in operation.operands
            )
            renamed_outputs[operation.out] = output
            namespaced_operations.append(
                dataclasses.replace(
                    operation,
                    operands=operands,
                    out=output,
                )
            )
        operations = namespaced_operations
        tensors = self._infer_vec_tensors(operations)
        uses: dict[str, list[int]] = {}
        for index, operation in enumerate(operations):
            for operand in operation.operands:
                if isinstance(operand, str):
                    uses.setdefault(operand, []).append(index)

        segments: list[tuple[int, int]] = []
        segment_start = 0
        segment_key = self._domain_key(tensors[operations[0].out])
        for index, operation in enumerate(operations[1:], start=1):
            key = self._domain_key(tensors[operation.out])
            if key != segment_key:
                segments.append((segment_start, index))
                segment_start = index
                segment_key = key
        segments.append((segment_start, len(operations)))

        operation_ids: list[str] = []
        broadcast_cache: dict[tuple[Any, ...], str] = {}
        preexisting_buffers = set(self.buffer_tensors)
        for start, end in segments:
            pointwise_id = f"pointwise_{self.pointwise_id}"
            self.pointwise_id += 1
            domain = tensors[operations[start].out]
            segment_outputs = {operation.out for operation in operations[start:end]}
            boundary_outputs = {
                operation.out
                for index, operation in enumerate(operations[start:end], start=start)
                if operation.out in preexisting_buffers
                or index == len(operations) - 1
                or any(consumer >= end for consumer in uses.get(operation.out, ()))
            }
            for output in boundary_outputs:
                self._add_allocation(output, tensors[output])

            rewritten: list[VecOp] = []
            inputs: set[str] = set()
            for operation in operations[start:end]:
                operands: list[Any] = []
                for operand in operation.operands:
                    rewritten_operand = operand
                    if isinstance(operand, str) and operand in tensors:
                        operand_tensor = tensors[operand]
                        if self._domain_key(operand_tensor) != self._domain_key(domain):
                            rewritten_operand = self._request_broadcast(
                                operand,
                                operand_tensor,
                                domain,
                                pointwise_id,
                                region_id,
                                broadcast_cache,
                            )
                        if rewritten_operand not in segment_outputs:
                            inputs.add(str(rewritten_operand))
                    operands.append(rewritten_operand)
                rewritten.append(
                    dataclasses.replace(
                        operation,
                        operands=tuple(operands),
                    )
                )

            vec_region, temp_allocations = KernelBufferPlanner().plan_vec_region(
                rewritten,
                local_shape=domain.shape_expr(block=self.block),
                default_dtype=domain.dtype,
                protected_buffers=set(self.allocated),
                buffer_shapes=self.buffer_shapes,
                buffer_dtypes=self.buffer_dtypes,
            )
            for allocation in temp_allocations:
                tensor = tensors.get(allocation.name)
                if tensor is None:
                    tensor = TileLangTensor.from_shape(
                        allocation.shape,
                        allocation.dtype,
                    )
                self._add_allocation(
                    allocation.name,
                    tensor,
                    shape=allocation.shape,
                )

            pointwise = PointwiseOpPlan(
                op_id=pointwise_id,
                inputs=tuple(sorted(inputs)),
                outputs=tuple(sorted(boundary_outputs)),
                origin=OperationOrigin("pointwise", pointwise_id, region_id),
                compute=vec_region,
            )
            self.operations.append(pointwise)
            self.operation_regions[pointwise_id] = region_id
            operation_ids.append(pointwise_id)
            for output in boundary_outputs:
                self.buffer_tensors[output] = tensors[output]
        return tuple(operation_ids)

    def _iteration(self, reduction_ops: tuple[str, ...]) -> RTileIterationPlan:
        return RTileIterationPlan(
            self.r_loop,
            "NO_LOOP" if self.kernel.persistent_reduction else "SERIAL",
            reduction_ops,
        )

    def _new_r_region(
        self,
        required_locs: set[str],
        operation_ids: tuple[str, ...],
        output_transfers: tuple[TransferPlan, ...] = (),
    ) -> None:
        region_id = f"r_tile_{self.region_id}"
        self.region_id += 1
        reduction_ops = tuple(
            op_id
            for op_id in operation_ids
            if isinstance(
                next(op for op in self.operations if op.op_id == op_id),
                ReductionOpPlan,
            )
        )
        self.region_drafts.append(
            RTileRegionPlan(
                region_id=region_id,
                loops=(self.r_loop,),
                block=self.block,
                allocations=(),
                input_transfers=self._input_transfers(required_locs),
                operations=operation_ids,
                output_transfers=output_transfers,
                iteration=self._iteration(reduction_ops),
            )
        )
        for op_id in operation_ids:
            self.operation_regions[op_id] = region_id

    def _plan_reductions(self) -> None:
        for accumulator, (result_var, dtype) in self.kernel._reduction_outputs.items():
            reduction_type, value, _ = self.kernel._reduction_vars[str(result_var)]
            region_id = f"r_tile_{self.region_id}"
            reduce_input = f"_{result_var}_reduce_in"
            source, vec_operations = self._raw_vec_ops(value, reduce_input, dtype)
            pointwise_ids = self._lower_vec_operations(
                vec_operations,
                region_id,
                reduce_input,
            )
            source_name = str(source)
            if source_name not in self.buffer_tensors:
                raise TileLangUnsupported(
                    "reduction-planning",
                    f"reduction source '{source_name}' has no tensor metadata",
                )
            reduction_id = f"reduction_{self.reduction_id}"
            self.reduction_id += 1
            source_tensor = self.buffer_tensors[source_name]
            reduction = ReductionOpPlan(
                op_id=reduction_id,
                inputs=(source_name,),
                outputs=(accumulator,),
                origin=OperationOrigin("reduction", str(result_var), reduction_type),
                source=source_name,
                source_tensor=source_tensor,
                accumulator=accumulator,
                reduction_type=reduction_type,
                reduce_dims=("r",),
                size=source_tensor.shape_list_expr(
                    block=self.block,
                    dim_extents={"r": self.r_loop.remain},
                ),
            )
            self.operations.append(reduction)
            operation_ids = (*pointwise_ids, reduction_id)
            required_locs = self._collect_input_locs(value)
            self._new_r_region(required_locs, operation_ids)

    def _plan_vector_outputs(self, tensor_arg_names: set[str]) -> None:
        vector_locs = self.kernel._valid_vector_epilogue_locs(tensor_arg_names)
        for out_loc, (result_var, dtype) in self.kernel._output_items():
            if out_loc not in vector_locs:
                continue
            region_id = f"r_tile_{self.region_id}"
            source, vec_operations = self._raw_vec_ops(result_var, out_loc, dtype)
            pointwise_ids = self._lower_vec_operations(
                vec_operations,
                region_id,
                out_loc,
            )
            if not pointwise_ids and str(source) != out_loc:
                copy_op = VecOp.create(
                    "copy", (str(source),), out_loc, dtype
                ).with_dtypes(dtype, dtype)
                pointwise_ids = self._lower_vec_operations(
                    (copy_op,),
                    region_id,
                    out_loc,
                )
            var = next(
                var
                for _, (var, loc, _) in self.kernel._output_buffer_items()
                if loc == out_loc
            )
            store = self._matrix_store_transfer(var, out_loc)
            required_locs = self._collect_input_locs(result_var)
            self._new_r_region(required_locs, pointwise_ids, (store,))

    def _plan_scalar_outputs(self, tensor_arg_names: set[str]) -> None:
        vector_locs = self.kernel._valid_vector_epilogue_locs(tensor_arg_names)
        for out_loc, (result_var, dtype) in self.kernel._output_items():
            if (
                out_loc in self.kernel._reduction_outputs
                or out_loc in vector_locs
                or not self.kernel._is_tensor_output_loc(out_loc, tensor_arg_names)
            ):
                continue
            region_id = f"once_{self.region_id}"
            self.region_id += 1
            source, vec_operations = self._raw_vec_ops(result_var, out_loc, dtype)
            operation_ids = self._lower_vec_operations(
                vec_operations,
                region_id,
                out_loc,
            )
            local_transfers: list[TransferPlan] = []
            if not operation_ids and str(source) != out_loc:
                local_transfers.append(TransferPlan("COPY", str(source), out_loc))
            var = next(
                var
                for _, (var, loc, _) in self.kernel._output_buffer_items()
                if loc == out_loc
            )
            tensor = self.buffer_tensors[out_loc]
            local_transfers.append(
                TransferPlan(
                    "COPY",
                    tensor.tiled_ref(out_loc, self.loops, local=True),
                    tensor.tiled_ref(var, self.loops, local=False),
                )
            )
            self.region_drafts.append(
                KernelRegionPlan(
                    region_id=region_id,
                    loops=(),
                    block=self.block,
                    allocations=(),
                    input_transfers=(),
                    operations=operation_ids,
                    output_transfers=tuple(local_transfers),
                )
            )

    def _plan_direct_reduction_stores(self, tensor_arg_names: set[str]) -> None:
        for _, (var, loc, _) in self.kernel._output_buffer_items():
            if loc not in self.kernel._reduction_outputs or var not in tensor_arg_names:
                continue
            tensor = self.buffer_tensors[loc]
            self.outer_output_transfers.append(
                TransferPlan(
                    "COPY",
                    tensor.tiled_ref(loc, self.loops, local=True),
                    tensor.tiled_ref(var, self.loops, local=False),
                )
            )

    def _assign_region_operations(
        self,
        graph: TileLangOperationGraph,
    ) -> tuple[KernelRegionPlan, ...]:
        operations_by_id = {
            operation.op_id: operation for operation in graph.operations
        }
        operations_by_region: dict[str, list[str]] = {
            region.region_id: [] for region in self.region_drafts
        }
        for operation in graph.operations:
            if isinstance(operation, BroadcastOpPlan):
                region_id = self.broadcast_regions[operation.destination]
            else:
                region_id = self.operation_regions[operation.op_id]
            operations_by_region[region_id].append(operation.op_id)
        return tuple(
            dataclasses.replace(
                region,
                operations=tuple(operations_by_region[region.region_id]),
                iteration=(
                    dataclasses.replace(
                        region.iteration,
                        reduction_ops=tuple(
                            op_id
                            for op_id in operations_by_region[region.region_id]
                            if isinstance(operations_by_id[op_id], ReductionOpPlan)
                        ),
                    )
                    if isinstance(region, RTileRegionPlan)
                    and region.iteration is not None
                    else getattr(region, "iteration", None)
                ),
            )
            if isinstance(region, RTileRegionPlan)
            else dataclasses.replace(
                region,
                operations=tuple(operations_by_region[region.region_id]),
            )
            for region in self.region_drafts
        )

    def build(self, name: Optional[str] = None) -> TileLangKernelPlan:
        argdefs, _, signature, _ = self.kernel.args.python_argdefs()
        tensor_arg_names = {
            argdef.name
            for argdef, sig in zip(argdefs, signature)
            if isinstance(sig, TensorArg)
        }
        self._register_inputs()
        self._register_outputs(tensor_arg_names)
        self._plan_reductions()
        self._plan_vector_outputs(tensor_arg_names)
        self._plan_scalar_outputs(tensor_arg_names)
        self._plan_direct_reduction_stores(tensor_arg_names)

        graph = BroadcastInsertionPass().run(
            self.operations,
            self.external_values,
            self.broadcast_requests,
        )
        subregions = self._assign_region_operations(graph)
        if self.ranked:
            axis_grid_vars = [
                f"_tl_axis{axis_index}_grid"
                for axis_index in range(len(self.pointwise_loops))
            ]
            prologue = [
                f"{grid_var} = T.ceildiv({axis_dim_var}, {axis_block_var})"
                for grid_var, axis_dim_var, axis_block_var in zip(
                    axis_grid_vars,
                    self.pointwise_axis_dim_vars,
                    self.pointwise_axis_block_vars,
                )
            ]
            for axis_index, axis_loop in enumerate(self.pointwise_loops):
                suffix_grids = axis_grid_vars[axis_index + 1:]
                suffix_product = " * ".join(suffix_grids)
                if suffix_product:
                    pid_expr = (
                        f"((cid // ({suffix_product})) % "
                        f"{axis_grid_vars[axis_index]})"
                    )
                else:
                    pid_expr = f"(cid % {axis_grid_vars[axis_index]})"
                prologue.extend((
                    f"{axis_loop.loop_var} = {pid_expr}",
                    f"{axis_loop.base} = {axis_loop.loop_var} * {axis_loop.block}",
                    f"{axis_loop.remain} = T.min({axis_loop.block}, "
                    f"{axis_loop.extent} - {axis_loop.base})",
                ))
            launch = KernelLaunchPlan(
                " * ".join(
                    f"T.ceildiv({axis_dim_var}, {axis_block_var})"
                    for axis_dim_var, axis_block_var in zip(
                        self.pointwise_axis_dim_vars,
                        self.pointwise_axis_block_vars,
                    )
                )
            )
            preamble = (
                f"_RBLOCK = {self.reduction_axis_name.upper()}BLOCK",
                *(
                    f"{axis_block_var} = {axis_name.upper()}BLOCK"
                    for axis_name, axis_block_var in zip(
                        self.pointwise_axis_names,
                        self.pointwise_axis_block_vars,
                    )
                ),
            )
            outer_loops = self.pointwise_loops
            body_prologue = ()
            serial_loop = None
        else:
            launch = KernelLaunchPlan("T.ceildiv(_xnumel, _XBLOCK)")
            preamble = (
                "_RBLOCK = R0BLOCK",
                "_XBLOCK = X0BLOCK",
                "_XBLOCK_SUB = X0BLOCK_SUB",
            )
            prologue = (
                "_tl_x_base = cid * _XBLOCK",
                "_tl_outer_remain = T.min(_XBLOCK, _xnumel - _tl_x_base)",
            )
            body_prologue = (
                "_tl_base = cid * _XBLOCK + _tl_block * _XBLOCK_SUB",
                "_remain_X = T.min(_XBLOCK_SUB, T.min("
                "_XBLOCK - _tl_block * _XBLOCK_SUB, _xnumel - _tl_base))",
            )
            outer_loops = (self.x_loop,)
            serial_loop = self.x_loop
        outer_region = XTileRegionPlan(
            region_id="x_tile_0",
            loops=outer_loops,
            block=self.block,
            allocations=tuple(self.allocations),
            input_transfers=tuple(self.outer_input_transfers),
            operations=(),
            output_transfers=tuple(self.outer_output_transfers),
            prologue=tuple(prologue),
            body_prologue=body_prologue,
            serial_loop=serial_loop,
            subregions=subregions,
        )
        return TileLangKernelPlan(
            signature=self._signature(name, tensor_arg_names),
            launch=launch,
            operations=graph.operations,
            regions=(outer_region,),
            preamble=preamble,
        )
