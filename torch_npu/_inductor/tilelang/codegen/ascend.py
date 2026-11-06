"""Pointwise VF emission for the upstream TileLang Ascend dialect.

This module consumes the typed plan; it does not import TileLang or access a
device. Intermediate values are scalar expressions inside one SIMT VF. Only
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


class AscendPointwiseEmitter:
    def __init__(self, code, allocations):
        self.code = code
        self.allocations = allocations

    @staticmethod
    def expression(name: str, operands: list[str]) -> str:
        binary = {"add": "+", "sub": "-", "mul": "*", "truediv": "/"}
        if name in binary:
            return f"({operands[0]} {binary[name]} {operands[1]})"
        if name == "copy":
            return operands[0]
        if name in {"exp", "log", "sqrt", "rsqrt", "abs"}:
            return f"T.{name}({operands[0]})"
        # min/max/activation NaN semantics need separate validation.
        raise NotImplementedError(f"A5 TileLang pointwise op is not enabled: {name}")

    def emit(self, operation) -> None:
        if not operation.outputs:
            raise NotImplementedError("A5 pointwise VF requires a boundary output")
        shape = buffer_shape(self.allocations[operation.outputs[0]].shape)
        if not shape:
            raise NotImplementedError("A5 pointwise VF requires a ranked tile")
        for output in operation.outputs:
            if buffer_shape(self.allocations[output].shape) != shape:
                raise NotImplementedError("A5 VF outputs require equal tile shapes")
        lanes = [f"_vf_i{i}" for i in range(len(shape))]

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
            return f"{name}[{', '.join(indices)}]"

        self.code.writeline("with T.SimtVF(threads=128):")
        with self.code.indent():
            self.code.writeline(
                f"for {', '.join(lanes)} in T.Parallel({', '.join(shape)}):"
            )
            with self.code.indent():
                values = {}
                for index, op in enumerate(operation.compute.operations):
                    execution_dtype = dtype_name(op.execution_dtype)
                    result_dtype = dtype_name(op.dtype)
                    if execution_dtype not in {"float16", "float32", "bfloat16"} or result_dtype not in {
                        "float16", "float32", "bfloat16"
                    }:
                        raise NotImplementedError("A5 pointwise currently supports floating-point values only")
                    # A scalar/broadcast intermediate is recomputed per lane;
                    # it needs no materialized expansion in UB.
                    reference(op.out)
                    operands = []
                    for operand in op.operands:
                        value = (
                            values[operand] if operand in values else reference(operand)
                        ) if isinstance(operand, str) else repr(operand)
                        operands.append(f"T.cast({value}, '{execution_dtype}')")
                    value = f"_vf_v{index}"
                    expression = self.expression(op.name, operands)
                    self.code.writeline(f"{value} = T.cast({expression}, '{result_dtype}')")
                    values[op.out] = value
                for output in operation.outputs:
                    self.code.writeline(f"{reference(output)} = {values[output]}")
