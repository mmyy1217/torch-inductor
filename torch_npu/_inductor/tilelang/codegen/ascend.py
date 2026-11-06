"""Pointwise VF emission for the upstream TileLang Ascend dialect.

This module consumes the typed plan; it does not import TileLang or access a
device. Intermediate values are vector registers inside one SIMD VF. Only
the operation's boundary outputs are written back to UB.
"""

from __future__ import annotations

import ast


def buffer_shape(shape: str) -> tuple[str, ...]:
    node = ast.parse(shape, mode="eval").body
    items = node.elts if isinstance(node, ast.Tuple) else [node]
    return tuple(ast.unparse(item) for item in items)


def dtype_name(dtype) -> str:
    return str(dtype).removeprefix("torch.")


def validate_simd_tile_width(width: int) -> None:
    if width <= 0 or width % 64:
        raise ValueError("A5 SIMD physical tile width must be a positive multiple of 64")


class AscendPointwiseEmitter:
    def __init__(self, code, allocations):
        self.code = code
        self.allocations = allocations

    @staticmethod
    def expression(name: str, operands: list[str]) -> str:
        instructions = {
            "add": "vadd", "sub": "vsub", "mul": "vmul", "truediv": "vdiv",
            "exp": "vexp", "log": "vln", "sqrt": "vsqrt", "abs": "vabs",
        }
        if name == "copy":
            return operands[0]
        if name == "rsqrt":
            return (
                "T.simd.vdiv(T.simd.vdup(1.0, 'float32'), "
                f"T.simd.vsqrt({operands[0]}, _vf_mask), _vf_mask)"
            )
        if name in instructions:
            return f"T.simd.{instructions[name]}({', '.join(operands)}, _vf_mask)"
        raise NotImplementedError(f"A5 TileLang pointwise op is not enabled: {name}")

    @staticmethod
    def rounded(value, dtype):
        # Keep registers in FP32 with 64 logical lanes. A narrow/widen pair
        # preserves the typed plan's low-precision rounding boundaries.
        if dtype == "float32":
            return value
        return (
            f"T.simd.vcvt(T.simd.vcvt({value}, '{dtype}', _vf_mask, "
            "sat=False), 'float32', _vf_mask)"
        )

    def emit(self, operation) -> None:
        if not operation.outputs:
            raise NotImplementedError("A5 pointwise VF requires a boundary output")
        shape = buffer_shape(self.allocations[operation.outputs[0]].shape)
        if not shape:
            raise NotImplementedError("A5 pointwise VF requires a ranked tile")
        for output in operation.outputs:
            if buffer_shape(self.allocations[output].shape) != shape:
                raise NotImplementedError("A5 VF outputs require equal tile shapes")
        # Unmasked vld reads a full vector. Only permit physical row widths
        # known to contain complete 64-element chunks, including our tile knob.
        width = shape[-1]
        if width != "_XBLOCK_SUB" and (not width.isdecimal() or int(width) % 64 or int(width) == 0):
            raise NotImplementedError("A5 SIMD requires a physical tile width divisible by 64")
        lanes = [f"_vf_i{i}" for i in range(len(shape) - 1)] + ["_vf_chunk * 64"]

        def reference(name):
            source_shape = buffer_shape(self.allocations[name].shape)
            if len(source_shape) > len(shape):
                raise NotImplementedError("A5 VF operand rank exceeds output rank")
            offset = len(shape) - len(source_shape)
            indices = []
            for i, extent in enumerate(source_shape):
                if extent == "1":
                    indices.append("0")
                elif extent == shape[offset + i]:
                    indices.append(lanes[offset + i])
                else:
                    raise NotImplementedError("A5 VF operand needs an explicit layout conversion")
            return f"{name}[{', '.join(indices)}]", source_shape[-1] == "1"

        def check_dtype(dtype):
            dtype = dtype_name(dtype)
            if dtype not in {"float16", "float32", "bfloat16"}:
                raise NotImplementedError("A5 pointwise currently supports floating-point values only")
            return dtype

        # Validate the plan before writing any code.
        for op in operation.compute.operations:
            check_dtype(op.execution_dtype)
            check_dtype(op.dtype)
            reference(op.out)
            self.expression(op.name, ["operand"] * len(op.operands))
            for operand in op.operands:
                if isinstance(operand, str):
                    reference(operand)
                    check_dtype(self.allocations[operand].dtype)

        self.code.writeline("with T.SimdVF():")
        with self.code.indent():
            self.code.writeline("_vf_mask = T.simd.pset(32)")
            self._emit_loops(operation, shape, lanes, reference, check_dtype)

    def _emit_loops(self, operation, shape, lanes, reference, check_dtype, depth=0):
        if depth < len(shape) - 1:
            self.code.writeline(f"for {lanes[depth]} in T.serial({shape[depth]}):")
            with self.code.indent():
                self._emit_loops(operation, shape, lanes, reference, check_dtype, depth + 1)
            return
        self.code.writeline(f"for _vf_chunk in T.serial(({shape[-1]}) // 64):")
        with self.code.indent():
            values = {}
            for index, op in enumerate(operation.compute.operations):
                execution_dtype = check_dtype(op.execution_dtype)
                result_dtype = check_dtype(op.dtype)
                operands = []
                for operand in op.operands:
                    if not isinstance(operand, str):
                        value = f"T.simd.vdup(T.cast({repr(operand)}, 'float32'), 'float32')"
                    elif operand in values:
                        value = values[operand]
                    else:
                        addr, broadcast = reference(operand)
                        dtype = check_dtype(self.allocations[operand].dtype)
                        dist = ("BRC_B32" if dtype == "float32" else "BRC_B16") if broadcast else (
                            "NORM" if dtype == "float32" else "UNPK_B16"
                        )
                        value = f"T.simd.vld({addr}, dist='{dist}')"
                        if dtype != "float32":
                            value = f"T.simd.vcvt({value}, 'float32', _vf_mask)"
                    operands.append(self.rounded(value, execution_dtype))
                value = f"_vf_v{index}"
                expression = self.expression(op.name, operands)
                # Arithmetic in the original typed expression executes before
                # the result cast (e.g. half multiply -> float32).
                expression = self.rounded(expression, execution_dtype)
                if result_dtype != execution_dtype:
                    expression = self.rounded(expression, result_dtype)
                self.code.writeline(f"{value} = {expression}")
                values[op.out] = value
            for output in operation.outputs:
                addr, _ = reference(output)
                dtype = check_dtype(self.allocations[output].dtype)
                value = values[output]
                dist = "NORM_B32"
                if dtype != "float32":
                    value = f"T.simd.vcvt({value}, '{dtype}', _vf_mask, sat=False)"
                    dist = "PK_B32"
                self.code.writeline(f"T.simd.vsts({addr}, {value}, _vf_mask, dist='{dist}')")
