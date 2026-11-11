"""Specialize the dense TileLang source during Inductor codegen."""

import ast
import json
from pathlib import Path


def render_flex_attention(spec):
    """Embed static shape/scale without importing TileLang in the codegen process."""
    config = json.loads(spec)
    template = (
        Path(__file__).resolve().parents[1] / "kernel/tilelang_flex_kernel.py"
    ).read_text()
    source = template + (
        f"\ninput_shape = {tuple(config['shape'])!r}\n"
        f"program = dense_attention_prim_func(input_shape, {config['scale']!r})\n"
    )
    ast.parse(source)
    # The wrapper embeds the complete source in a triple-single-quoted literal.
    if "'" * 3 in source or "\\" in source:
        raise ValueError("TileLang source is unsafe for wrapper string embedding")
    return source
