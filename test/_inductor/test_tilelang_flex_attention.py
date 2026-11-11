"""CPU regressions for dense input contracts and template lifecycle.

Run with matching torch/torch_npu 2.14. Device compilation and launch are mocked;
these tests check compiler plumbing, not NPU numerical correctness.
"""

import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
import tempfile
import threading
from concurrent.futures import Future
from unittest.mock import patch
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401 - Register the NPU backend.
from torch.nn.attention.flex_attention import (
    flex_attention as torch_flex_attention,
    create_block_mask,
    AuxRequest,
)
from torch._inductor.lowering import lowerings
import torch._inductor.kernel.flex.flex_attention
from torch._inductor.codegen.cpp import CppScheduling
from torch._inductor.utils import run_and_get_code

ROOT = Path(__file__).resolve().parents[2]


def load_source(name, relative):
    """Load checkout code over the installed wheel for the CPU tests below."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


renderer = load_source(
    "torch_npu._inductor.codegen.tilelang_template",
    "torch_npu/_inductor/codegen/tilelang_template.py",
)
cg = load_source(
    "torch_npu._inductor.codegen.tilelang", "torch_npu/_inductor/codegen/tilelang.py"
)
rt = load_source(
    "torch_npu._inductor.runtime.tilelang", "torch_npu/_inductor/runtime/tilelang.py"
)
lower = load_source(
    "torch_npu._inductor.kernel.tilelang_flex_attention",
    "torch_npu/_inductor/kernel/tilelang_flex_attention.py",
)


def flex_attention(q, k, v, **kwargs):
    """Supply the Python entry's dense flag while CPU tests bypass NPU initialization."""
    options = dict(kwargs.pop("kernel_options", {}))
    options["_NPU_TILELANG_CACHE_KEY"] = cg.template_cache_key()
    options["_NPU_DENSE_ATTENTION"] = (
        kwargs.get("score_mod") is None and kwargs.get("block_mask") is None
    )
    aux = kwargs.get("return_aux")
    options["_NPU_EXPLICIT_AUX"] = kwargs.get("return_lse", False) or (
        aux is not None and (aux.lse or aux.max_scores)
    )
    return torch_flex_attention(q, k, v, kernel_options=options, **kwargs)


def reference(q, k, v, scale=None):
    """Dense FP32 reference used by mocked launches and dense assertions."""
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    group = q.shape[-3] // k.shape[-3]
    k = k.repeat_interleave(group, dim=-3)
    v = v.repeat_interleave(group, dim=-3)
    return (
        torch.softmax((q.float() @ k.float().transpose(-2, -1)) * scale, -1) @ v.float()
    ).to(q.dtype)


def inputs():
    """Create deterministic BF16 CPU inputs shared by template tests."""
    torch.manual_seed(2026)
    return [torch.randn(1, 4, 256, 128, dtype=torch.bfloat16) for _ in range(3)]


class FlexAttentionTests(unittest.TestCase):
    @contextlib.contextmanager
    def template_path(self):
        """Run real lowering and wrapper generation with a CPU stand-in for the NPU kernel."""
        old_codegen = CppScheduling.codegen_template
        self.buffers = []
        self.calls = []
        self.launch_args = []
        self.compilations = []
        source_specs = {}

        def codegen(scheduling, node, epilogue, prologue=()):
            """Route template nodes through real codegen while retaining CPU scheduling."""
            if cg.is_tilelang_template(node):
                self.buffers.append(node.node)
                self.deps = node.read_writes
                source_specs[renderer.render_flex_attention(node.node.spec).strip()] = (
                    node.node.spec
                )
                # CPU scheduling exercises real wrapper generation, with an explicit
                # device ordinal supplied only at the TileLang codegen boundary.
                with patch.object(
                    node.node, "get_device", lambda: torch.device("cpu", 0)
                ):
                    return cg.codegen_tilelang_template(
                        scheduling, node, epilogue, prologue
                    )
            return old_codegen(scheduling, node, epilogue, prologue)

        def launch(q, k, v, *, out, spec):
            """Use CPU dense math to verify shape/scale plumbing, not device numerics."""
            self.calls.append(out.data_ptr())
            self.launch_args.append(spec)
            out.copy_(reference(q, k, v, json.loads(spec)["scale"]))

        def compile_kernel(source, device_index):
            """Record compilation and return the CPU launcher used by template_path."""
            self.compilations.append((source, device_index))
            spec = source_specs[source.strip()]
            return SimpleNamespace(
                launch=lambda q, k, v, *, out: launch(q, k, v, out=out, spec=spec)
            )

        torch._dynamo.reset()
        with (
            torch._inductor.config.patch(force_disable_caches=True),
            patch.dict(
                lowerings,
                {
                    torch.ops.higher_order.flex_attention: lower.lower_tilelang_flex_attention
                },
            ),
            patch.object(lower, "check_device", lambda device: None),
            patch.object(CppScheduling, "codegen_template", codegen),
            patch.object(rt, "compile_tilelang", compile_kernel),
        ):
            yield

    def test_template_compile_and_reuse(self):
        """Check template IR, wrapper allocation and reuse without runtime recompilation."""
        xs = inputs()
        with self.template_path(), torch.no_grad():
            fn = torch.compile(
                lambda q, k, v: flex_attention(q, k, v),
                backend="inductor",
                fullgraph=True,
            )
            y, codes = run_and_get_code(fn, *xs)
            torch.testing.assert_close(y, reference(*xs))
            self.assertEqual(self.calls, [y.data_ptr()])
            with (
                patch.object(
                    renderer,
                    "render_flex_attention",
                    side_effect=AssertionError("render during call"),
                ),
                patch.object(
                    rt,
                    "compile_tilelang",
                    side_effect=AssertionError("compile during call"),
                ),
            ):
                again = fn(*xs)
            torch.testing.assert_close(again, reference(*xs))
            self.assertEqual(self.calls, [y.data_ptr(), again.data_ptr()])
        self.assertEqual(len(self.buffers), 1)
        self.assertEqual(len(self.buffers[0].inputs), 3)
        buf = self.buffers[0]
        from torch._inductor.ir import TemplateBuffer, TritonTemplateBuffer

        self.assertIsInstance(buf, TemplateBuffer)
        self.assertNotIsInstance(buf, TritonTemplateBuffer)
        self.assertFalse(buf.allow_epilogue_fusion)
        self.assertFalse(buf.allow_prologue_fusion)
        deps = self.deps
        self.assertGreaterEqual(len(deps.reads), 3)
        self.assertEqual(len(deps.writes), 1)
        code = "\n".join(codes)
        self.assertIn("T.Pipelined(", code)
        self.assertIn("num_stages=3", code)
        self.assertNotIn("T.SimtVF", code)
        self.assertNotIn("FullCounts", code)
        self.assertIn("compile_tilelang(", code)
        self.assertIn(".launch(", code)
        self.assertLess(code.index("compile_tilelang("), code.index("def call("))
        self.assertEqual(len(self.compilations), 1)
        self.assertIn("out=buf", code)
        # The kernel ABI takes rank-4 Q/K/V/O and aliases them for the body.
        self.assertIn("Q4: T.Buffer((B, H, q_len, D), dtype)", code)
        self.assertIn("T.view(Q4, (BH, q_len, D))", code)
        self.assertNotIn("flex_tilelang.dense", code)
        self.assertNotIn("async_compile.triton(", code)

    def test_grad_enabled_inference(self):
        """Unused internal LSE requests must not reject inputs without gradients."""
        import importlib

        module = importlib.import_module("torch.nn.attention.flex_attention")
        original = module._apply_kernel_options

        def npu_options(*args, **kwargs):
            options = original(*args, **kwargs)
            options["OUTPUT_LOGSUMEXP"] = True
            return options

        with (
            self.template_path(),
            torch.enable_grad(),
            patch.object(module, "_apply_kernel_options", npu_options),
        ):
            xs = inputs()
            fn = torch.compile(lambda q, k, v: flex_attention(q, k, v), fullgraph=True)
            torch.testing.assert_close(fn(*xs), reference(*xs))

    def test_shared_specialization_registers_once_and_launches_twice(self):
        """Two identical attention specializations must share compilation, not output storage."""
        xs = inputs()
        other = [x + 0.25 for x in xs]
        with self.template_path(), torch.no_grad():
            fn = torch.compile(
                lambda q, k, v, q2, k2, v2: (
                    flex_attention(q, k, v),
                    flex_attention(q2, k2, v2),
                ),
                backend="inductor",
                fullgraph=True,
            )
            outputs, codes = run_and_get_code(fn, *xs, *other)
            torch.testing.assert_close(outputs[0], reference(*xs))
            torch.testing.assert_close(outputs[1], reference(*other))
        self.assertEqual(len(self.buffers), 2)
        self.assertEqual(len(self.compilations), 1)
        self.assertEqual(len(self.calls), 2)
        self.assertNotEqual(outputs[0].data_ptr(), outputs[1].data_ptr())
        code = "\n".join(codes)
        self.assertEqual(code.count("compile_tilelang("), 1)
        self.assertEqual(code.count(".launch("), 2)

    def test_reject_unsupported(self):
        """Reject score/mask customization, captured tensors, auxiliary outputs and training."""
        xs = inputs()
        captured = torch.arange(4, dtype=torch.float32)
        mask = create_block_mask(
            lambda b, h, m, n: m >= n,
            B=None,
            H=None,
            Q_LEN=256,
            KV_LEN=256,
            device="cpu",
        )
        funcs = [
            lambda q, k, v: flex_attention(
                q, k, v, score_mod=lambda s, b, h, m, n: s + 1
            ),
            lambda q, k, v: flex_attention(q, k, v, block_mask=mask),
            lambda q, k, v: flex_attention(
                q, k, v, score_mod=lambda s, b, h, m, n: s + captured[h]
            ),
            lambda q, k, v: flex_attention(q, k, v, return_aux=AuxRequest(lse=True)),
        ]
        for fn in funcs:
            with self.subTest(fn=fn), self.template_path(), torch.no_grad():
                with self.assertRaisesRegex(Exception, "TileLang FlexAttention v[12]"):
                    torch.compile(fn, backend="inductor", fullgraph=True)(*xs)
        with self.template_path():
            with self.assertRaisesRegex(Exception, "TileLang FlexAttention v[12]"):
                torch.compile(
                    lambda q, k, v: flex_attention(q, k, v),
                    backend="inductor",
                    fullgraph=True,
                )(*(x.requires_grad_() for x in inputs()))

    def test_reject_unsupported_input_layouts(self):
        """Reject FP16, sequence tails and noncontiguous inputs before device compilation."""
        variants = [
            [x.to(torch.float16) for x in inputs()],
            [x[:, :, :129, :].contiguous() for x in inputs()],
            [x.transpose(-2, -1) for x in inputs()],
            [x[..., :64].contiguous() for x in inputs()],
        ]
        for xs in variants:
            with (
                self.subTest(shape=xs[0].shape, dtype=xs[0].dtype),
                self.template_path(),
                torch.no_grad(),
            ):
                with self.assertRaisesRegex(Exception, "TileLang FlexAttention v2"):
                    torch.compile(
                        lambda q, k, v: flex_attention(q, k, v),
                        backend="inductor",
                        fullgraph=True,
                    )(*xs)

    def test_parameterized_shape_and_scale(self):
        """Verify shape and scale reach the generated specialization instead of staying fixed."""
        shapes = [
            (2, 2, 2, 128, 256, 128),
            (1, 3, 3, 512, 128, 256),
            (2, 8, 2, 128, 256, 256),
            (2, 4, 1, 256, 128, 512),
        ]
        for b, h, hk, m, n, d in shapes:
            with self.subTest(shape=(b, h, hk, m, n, d)):
                torch.manual_seed(2026)
                q = torch.randn(b, h, m, d, dtype=torch.bfloat16)
                k = torch.randn(b, hk, n, d, dtype=q.dtype)
                v = torch.randn_like(k)
                with self.template_path(), torch.no_grad():
                    fn = torch.compile(
                        lambda q, k, v: flex_attention(
                            q, k, v, scale=0.125, enable_gqa=True
                        ),
                        backend="inductor",
                        fullgraph=True,
                        dynamic=False,
                    )
                    actual = fn(q, k, v)
                    torch.testing.assert_close(actual, reference(q, k, v, scale=0.125))
                    self.assertEqual(tuple(actual.shape), tuple(q.shape))
                    spec = json.loads(self.launch_args[0])
                    self.assertEqual(spec["shape"], [b, h, m, n, d, hk])

    def test_cache_keys(self):
        """Template fingerprints in real FX graphs must separate both PyTorch disk caches."""
        from torch._inductor.codecache import compiled_fx_graph_hash
        from torch._functorch._aot_autograd.autograd_cache import (
            autograd_cache_key,
            sanitize_gm_for_cache,
        )
        from torch._functorch._aot_autograd.schemas import AOTConfig

        with patch.object(rt, "compiler_fingerprint", return_value="toolchain-A"):
            version_a = cg.template_cache_key()
        with patch.object(rt, "compiler_fingerprint", return_value="toolchain-B"):
            version_b = cg.template_cache_key()
        self.assertNotEqual(version_a, version_b)

        def capture(gm, example_inputs):
            # Use one real graph so only the version constant changes between hashes.
            node = next(
                n
                for n in gm.graph.nodes
                if n.target is torch.ops.higher_order.flex_attention
            )
            self.assertEqual(node.args[6]["_NPU_TILELANG_CACHE_KEY"], version_a)
            aot = AOTConfig(None, None, None, None, 0, 0, False)
            keys = []
            for version in (version_a, version_a, version_b):
                node.update_arg(6, dict(node.args[6], _NPU_TILELANG_CACHE_KEY=version))
                gm.recompile()
                with sanitize_gm_for_cache(gm):
                    fx_key = compiled_fx_graph_hash(gm, example_inputs, {}, [])[0]
                keys.append((fx_key, autograd_cache_key(gm, example_inputs, aot)[0]))
            self.assertEqual(keys[0], keys[1])
            self.assertNotEqual(keys[0][0], keys[2][0])
            self.assertNotEqual(keys[0][1], keys[2][1])
            return lambda *args: (args[0],)

        torch._dynamo.reset()
        with (
            patch.object(rt, "compiler_fingerprint", return_value="toolchain-A"),
            torch.no_grad(),
        ):
            torch.compile(
                flex_attention, backend=capture, fullgraph=True, dynamic=False
            )(*inputs())


class TemplateLifecycleTests(unittest.TestCase):
    """Exercise actual compile registration/cache/launch with only the DSL mocked."""

    def setUp(self):
        """Isolate runtime cache and diagnostic files for each lifecycle test."""
        artifact_root = os.environ.get("FLEX_TILELANG_TEST_ARTIFACT_DIR")
        if artifact_root is not None:
            Path(artifact_root).mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=artifact_root)
        self.addCleanup(self.tmp.cleanup)
        rt._compile_tilelang.cache_clear()
        self.addCleanup(rt._compile_tilelang.cache_clear)

    def test_compile_cache_reload_and_errors(self):
        """Check asynchronous compilation, cache invalidation, wrapper reload and errors."""
        from torch._inductor.codecache import PyCodeCache
        from torch._inductor.async_compile import AsyncCompile

        source = (
            "from types import SimpleNamespace\n"
            "program = SimpleNamespace(script=lambda: 'prim')\n"
            "input_shape = (1,4,256,256,128,4)\n"
        )
        state = threading.local()
        compiled = []

        @contextlib.contextmanager
        def device(index):
            """Track the selected device inside the mocked asynchronous compiler thread."""
            state.index = index
            try:
                yield
            finally:
                del state.index

        def compile_program(program, **kwargs):
            """Record compiler calls without importing TileLang or accessing an NPU."""
            compiled.append((state.index, threading.get_ident(), program.script()))
            return SimpleNamespace(
                get_kernel_source=lambda: "device",
                artifact=SimpleNamespace(
                    device_mod=SimpleNamespace(script=lambda: "lowered")
                ),
            )

        fake = SimpleNamespace(compile=compile_program)
        with (
            patch.dict(sys.modules, {"tilelang": fake}),
            patch.object(torch.npu, "device", device),
            patch.object(rt, "compiler_fingerprint", lambda: "toolchain-A"),
            patch.dict(os.environ, {"FLEX_TILELANG_ARTIFACT_DIR": self.tmp.name}),
            torch._inductor.config.patch(compile_threads=2),
        ):
            scope = {"kernel": rt.compile_tilelang(source, 0)}
            self.assertIsInstance(scope["kernel"], Future)
            AsyncCompile().wait(scope)
            first = scope["kernel"]
            self.assertNotEqual(compiled[0][1], threading.get_ident())
            self.assertIs(rt._compile_tilelang(source, 0, "toolchain-A"), first)
            rt._compile_tilelang(source + "# changed\n", 0, "toolchain-A")
            rt._compile_tilelang(source, 1, "toolchain-A")
            rt._compile_tilelang(source, 0, "toolchain-B")
            self.assertEqual(len(compiled), 4)
            self.assertEqual(compiled[2][0], 1)
            # Reloading a cached Python wrapper must rerun registration even
            # after process-local kernel/module caches have been cleared.
            wrapper = (
                "from torch_npu._inductor.runtime.tilelang import compile_tilelang\n"
                "from torch._inductor.async_compile import AsyncCompile\n"
                f"kernel = compile_tilelang({source!r}, 0)\n"
                "AsyncCompile().wait(globals())\n"
            )
            key, path = PyCodeCache.write(wrapper)
            rt._compile_tilelang.cache_clear()
            PyCodeCache.modules_no_attr.pop(path, None)
            loaded = PyCodeCache.load_by_key_path(key, path)
            self.assertIsInstance(loaded.kernel, rt.PreparedTileLangKernel)
            self.assertEqual(len(compiled), 5)
            with patch.object(
                fake, "compile", side_effect=RuntimeError("compile failed")
            ):
                failed = {"kernel": rt.compile_tilelang(source + "# fail", 0)}
                with self.assertRaisesRegex(RuntimeError, "compile failed"):
                    AsyncCompile().wait(failed)

    def test_prepared_launch_never_builds_or_compiles(self):
        """Repeated dense launches must only execute and preserve output ownership."""
        xs = inputs()
        calls = []

        def kernel(q, k, v, *args):
            """Stand in for the compiled kernel to check the supplied output buffer."""
            calls.append(args[-1].data_ptr())
            args[-1].copy_(reference(q, k, v))

        prepared = rt.PreparedTileLangKernel(
            kernel,
            (1, 4, 256, 256, 128, 4),
            None,
            Path(self.tmp.name),
        )
        with (
            patch.object(rt, "check_inputs", lambda *args: None),
            patch.object(torch.npu, "device", lambda _: contextlib.nullcontext()),
            patch.object(
                rt, "compile_tilelang", side_effect=AssertionError("runtime compile")
            ),
            patch.object(
                rt, "_compile_tilelang", side_effect=AssertionError("runtime compile")
            ),
            patch.object(
                renderer,
                "render_flex_attention",
                side_effect=AssertionError("runtime render"),
            ),
        ):
            for _ in range(2):
                out = torch.empty_like(xs[0])
                prepared.launch(*xs, out=out)
                torch.testing.assert_close(out, reference(*xs))
                self.assertEqual(calls[-1], out.data_ptr())
            self.assertEqual(len(calls), 2)
            with self.assertRaisesRegex(ValueError, "must not alias"):
                prepared.launch(*xs, out=xs[0])

    def test_rank4_launch_abi_and_contract(self):
        """Launch must pass the caller's rank-4 tensors through unchanged."""
        xs = inputs()
        received = []

        def kernel(*args):
            """Record the exact objects the compiled kernel would receive."""
            received.append(args)

        prepared = rt.PreparedTileLangKernel(
            kernel,
            (1, 4, 256, 256, 128, 4),
            None,
            Path(self.tmp.name),
        )
        with (
            patch.object(rt, "check_inputs", lambda *args: None),
            patch.object(torch.npu, "device", lambda _: contextlib.nullcontext()),
        ):
            out = torch.empty_like(xs[0])
            prepared.launch(*xs, out=out)
            self.assertEqual(len(received), 1)
            self.assertEqual(len(received[0]), 4)
            for passed, expected in zip(received[0], (*xs, out)):
                self.assertIs(passed, expected)
            self.assertEqual(received[0][3].data_ptr(), out.data_ptr())
            with self.assertRaisesRegex(ValueError, "must not alias"):
                prepared.launch(*xs, out=xs[0])
            with self.assertRaisesRegex(ValueError, "output contract mismatch"):
                prepared.launch(*xs, out=torch.empty(1, dtype=xs[0].dtype))
            self.assertEqual(prepared.launches, 1)
        # The real contract still rejects shape, dtype and stride violations.
        strided = torch.empty(1, 4, 512, 128)[:, :, ::2, :]
        self.assertEqual(tuple(strided.shape), tuple(xs[0].shape))
        self.assertFalse(strided.is_contiguous())
        rejects = (
            (xs[0][:, :2], xs[1], xs[2], torch.empty_like(xs[0][:, :2])),
            (xs[0].float(), xs[1], xs[2], torch.empty_like(xs[0])),
            (strided, xs[1], xs[2], torch.empty_like(xs[0])),
        )
        for q, k, v, out in rejects:
            with self.assertRaisesRegex(NotImplementedError, "contiguous BF16"):
                prepared.launch(q, k, v, out=out)
        self.assertEqual(prepared.launches, 1)


if __name__ == "__main__":
    unittest.main()
