"""Inductor TileLang compile registration, cache, and prepared-kernel launch ABI."""

from functools import lru_cache
from threading import RLock
import hashlib
import json
import os
import shutil
from pathlib import Path
import torch


def check_inputs(q, k, v, shapes):
    """Validate Q/K/V against the precomputed rank-4 ``(q_shape, kv_shape)``."""
    q_shape, kv_shape = shapes
    for x, shape in ((q, q_shape), (k, kv_shape), (v, kv_shape)):
        if (
            tuple(x.shape) != shape
            or x.dtype != torch.bfloat16
            or not x.is_contiguous()
        ):
            raise NotImplementedError(
                f"TileLang requires contiguous BF16 inputs; expected {shape}"
            )
        if torch.is_grad_enabled() and x.requires_grad:
            raise NotImplementedError("TileLang dense v1 is inference-only")
        if x.device.type != "npu":
            raise NotImplementedError("TileLang dense v1 requires Ascend NPU")
    if not (q.device == k.device == v.device):
        raise ValueError("Q/K/V must be on the same device")


def artifact_dir():
    path = Path(os.environ["FLEX_TILELANG_ARTIFACT_DIR"]).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def compiler_fingerprint():
    import importlib.metadata

    versions = []
    for package in ("tilelang", "tilelang-ascend", "torch", "torch-npu"):
        try:
            versions.append((package, importlib.metadata.version(package)))
        except importlib.metadata.PackageNotFoundError:
            versions.append((package, None))
    try:
        import tilelang
    except ModuleNotFoundError as error:
        if error.name != "tilelang":
            raise
        loaded = None
    else:
        loaded = (
            getattr(tilelang, "__file__", None),
            getattr(tilelang, "__version__", None),
        )
    environment = tuple(
        (name, os.environ.get(name))
        for name in (
            "ASCEND_HOME_PATH",
            "ASCEND_OPP_PATH",
            "BISHENG_HOME",
            "ASCEND_NPU_ARCH",
            "TILELANG_CACHE_DIR",
        )
    )
    compiler = shutil.which("bisheng")
    compiler = (compiler, str(Path(compiler).resolve())) if compiler else None
    return repr((versions, loaded, environment, compiler, "ascend", False))


class PreparedTileLangKernel:
    """A compiled callable with static ABI; launch never renders or compiles."""

    def __init__(self, kernel, shape, device_index, path):
        self.kernel = kernel
        self.shape = tuple(shape)
        self.device_index = device_index
        self.path = path
        self.launches = 0
        b, h, m, n, d, hk = self.shape
        # The compiled kernel takes the caller's rank-4 tensors; no reshape.
        self.input_shapes = ((b, h, m, d), (b, hk, n, d))
        self.out_shape = (b, h, m, d)

    def launch(self, q, k, v, *, out):
        check_inputs(q, k, v, self.input_shapes)
        if q.device.index != self.device_index:
            raise ValueError("TileLang compiled device does not match input device")
        if (
            out.shape != self.out_shape
            or out.dtype != q.dtype
            or out.device != q.device
            or not out.is_contiguous()
            or out.requires_grad
        ):
            raise ValueError("TileLang output contract mismatch")
        out_start = out.data_ptr()
        out_end = out_start + out.nbytes
        if any(
            max(out_start, x.data_ptr()) < min(out_end, x.data_ptr() + x.nbytes)
            for x in (q, k, v)
        ):
            raise ValueError("TileLang output must not alias inputs")
        with torch.npu.device(self.device_index):
            self.kernel(q, k, v, out)
        self.launches += 1
        if self.path is not None and self.launches == 1:
            (self.path / "runtime.json").write_text(
                json.dumps(
                    dict(
                        implementation="tilelang",
                        launched=True,
                        shape=tuple(q.shape),
                        kernel_launches_per_call=1,
                        device=str(q.device),
                    ),
                    indent=2,
                )
            )


_COMPILE_LOCK = RLock()


@lru_cache(maxsize=128)
def _compiled_program(source, device_index, fingerprint):
    import tilelang
    from torch._inductor.codecache import PyCodeCache

    with torch.npu.device(device_index):
        module = PyCodeCache.load(source, extra=repr((device_index, fingerprint)))
        kernel = tilelang.compile(module.program, target="ascend", out_idx=None)
    return module, kernel


@lru_cache(maxsize=128)
def _prepared_kernel(source, device_index, fingerprint, root):
    module, kernel = _compiled_program(source, device_index, fingerprint)
    path = None
    if root is not None:
        import tilelang

        key = hashlib.sha256(
            (source + repr(device_index) + fingerprint).encode()
        ).hexdigest()
        path = root / key
        path.mkdir(parents=True, exist_ok=True)
        # A reused directory must not claim this registration already launched.
        (path / "runtime.json").unlink(missing_ok=True)
        (path / "rendered_kernel.py").write_text(source)
        (path / "prim_func.py").write_text(module.program.script())
        (path / "device_code.txt").write_text(kernel.get_kernel_source())
        with torch.npu.device(device_index):
            artifact = getattr(kernel, "artifact", None)
            if artifact is None or getattr(artifact, "device_mod", None) is None:
                artifact = tilelang.lower(module.program, target="ascend")
            (path / "lowered_device_ir.py").write_text(artifact.device_mod.script())
        (path / "compiler.json").write_text(
            json.dumps(
                dict(
                    schema_version=2,
                    cache_key=key,
                    source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    fingerprint=fingerprint,
                    device_index=device_index,
                    target="ascend",
                    out_idx=None,
                    fast_math=False,
                    compilation_phase="wrapper_initialization",
                ),
                indent=2,
            )
        )
    return PreparedTileLangKernel(kernel, module.input_shape, device_index, path)


def _compile_tilelang(source, device_index, fingerprint):
    root = artifact_dir() if os.environ.get("FLEX_TILELANG_ARTIFACT_DIR") else None
    return _prepared_kernel(source, device_index, fingerprint, root)


def _clear_compile_cache():
    _prepared_kernel.cache_clear()
    _compiled_program.cache_clear()


_compile_tilelang.cache_clear = _clear_compile_cache


def compile_tilelang(source, device_index):
    """Submit template compilation to Inductor's normal compile/wait lifecycle."""
    from functools import partial
    from torch._inductor.async_compile import AsyncCompile

    return AsyncCompile.submit(
        partial(_compile_tilelang_locked, source, device_index, compiler_fingerprint())
    )


def _compile_tilelang_locked(source, device_index, fingerprint):
    # lru_cache alone allows concurrent misses to compile the same program twice.
    # Serialize compilation: Ascend compilation also mutates process-wide state.
    with _COMPILE_LOCK:
        return _compile_tilelang(source, device_index, fingerprint)
