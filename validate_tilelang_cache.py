"""Check cold compilation and cross-process FX cache replay on an allocated NPU."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--phase", choices=("cold", "warm"), required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
    parser.error("set ASCEND_RT_VISIBLE_DEVICES to your allocated device")
root = args.output.resolve()
if args.phase == "cold" and root.exists() and any(root.iterdir()):
    parser.error("cold output must be empty; preserve prior evidence")
if args.phase == "warm" and not (root / "cache-cold.json").is_file():
    parser.error("run cold phase first with the same output directory")
root.mkdir(parents=True, exist_ok=True)
os.environ.update(
    TORCHINDUCTOR_FORCE_DISABLE_CACHES="0",
    TORCHINDUCTOR_FX_GRAPH_CACHE="1",
    TORCHINDUCTOR_COMPILE_THREADS="1",
    TORCHINDUCTOR_CACHE_DIR=str(root / "inductor-cache"),
    TILELANG_CACHE_DIR=str(root / "tilelang-cache"),
    TRITON_CACHE_DIR=str(root / "triton-cache"),
    FLEX_TILELANG_ARTIFACT_DIR=str(root / "artifacts"),
)
import torch
import torch_npu  # noqa: F401 - Register the NPU backend.
import torch._functorch.config as functorch_config
import torch._inductor.config as inductor_config
from torch._dynamo.utils import counters
from torch_npu._inductor import config as npu_config
from torch_npu._inductor.codegen import tilelang_template

spec = importlib.util.spec_from_file_location(
    "matrix", Path(__file__).with_name("validate_tilelang_matrix.py")
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
npu_config.flex_attention.implementation = "tilelang"
functorch_config.enable_autograd_cache = False
inductor_config.fx_graph_cache = True
inductor_config.force_disable_caches = False
inductor_config.fx_graph_remote_cache = False
torch.npu.set_device(0)
case = m.build_case("full_only", "npu")
reference = m.reference(case)
phase = args.phase
original = tilelang_template.render_flex_attention
renders = []


def render(spec):
    renders.append(1)
    if phase == "warm":
        raise AssertionError("Warm FX cache replay unexpectedly rendered template")
    return original(spec)


with patch.object(tilelang_template, "render_flex_attention", render), torch.no_grad():
    fn = torch.compile(
        case["attention"],
        backend="inductor",
        fullgraph=True,
        dynamic=False,
        options={"npu_backend": "tilelang"},
    )
    actual = fn(*case["inputs"])
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu().float(), reference, atol=0.03, rtol=0.03)
    import tilelang

    with patch.object(
        tilelang, "compile", side_effect=AssertionError("Unexpected repeat compile")
    ):
        repeated = fn(*case["inputs"])
        torch.npu.synchronize()
    torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
if phase == "warm":
    assert counters["inductor"]["fxgraph_cache_hit"] > 0, "No FX graph cache hit"
else:
    assert renders, "Cold phase did not render a TileLang template"
result = dict(
    status="PASS",
    phase=phase,
    render_calls=len(renders),
    counters={k: dict(v) for k, v in counters.items()},
    max_abs_error=float((actual.cpu().float() - reference).abs().max()),
)
(root / f"cache-{phase}.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
