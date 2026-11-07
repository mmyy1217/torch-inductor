"""Run only inside npu-guard; validates the isolated A5 migration environment."""
import functools
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
import traceback

root = Path('/home/yja/experiments/tilelang-backend-a5-20261106')
evidence = root / os.environ.get('A5_REDUCTION_EVIDENCE', 'evidence-reduction')
evidence.mkdir(exist_ok=True)
import torch
import torch_npu

assert torch.__version__ == '2.14.0+cpu', torch.__version__
assert Path(torch_npu.__file__).resolve().is_relative_to(root / 'env'), torch_npu.__file__
# Select the runtime context before configuring its limits; no tensors yet.
torch.npu.set_device(0)
torch_npu.npu.set_device_limit(0, cube_num=32, vector_num=64)
limits = torch_npu.npu.get_device_limit(0)
assert limits['cube_core_num'] == 32 and limits['vector_core_num'] == 64, limits
torch.manual_seed(0)
import tilelang
import tilelang.ascend.language as T
assert Path(tilelang.__file__).resolve().is_relative_to(root / 'tilelang'), tilelang.__file__
metadata = {
    'torch': torch.__version__, 'torch_file': torch.__file__,
    'torch_npu': torch_npu.__version__, 'torch_npu_file': torch_npu.__file__,
    'tilelang_file': tilelang.__file__, 'device': 0, 'limits': limits,
    'ascend_home': os.environ.get('ASCEND_HOME_PATH'), 'seed': 0,
    'tilelang_sha': subprocess.check_output(['git','-C',str(root/'tilelang'),'rev-parse','HEAD'],text=True).strip(),
    'pytorch_sha': subprocess.check_output(['git','-C',str(root/'pytorch'),'rev-parse','HEAD'],text=True).strip(),
}
metadata['python_overlay'] = json.loads((root/'python-overlay.json').read_text()) if (root/'python-overlay.json').exists() else None
print(json.dumps(metadata,indent=2),flush=True)
(evidence/'environment.json').write_text(json.dumps(metadata,indent=2))
original_compile = tilelang.compile
compiles = []
@functools.wraps(original_compile)
def capture(func, *args, **kwargs):
    index = len(compiles)
    compiles.append(kwargs)
    (evidence/f'kernel-{index}.tir').write_text(func.script(show_meta=True))
    result = original_compile(func,*args,**kwargs)
    (evidence/f'kernel-{index}.cpp').write_text(result.kernel_source)
    assert '__simt_vf__' not in result.kernel_source, result.kernel_source
    assert '__simd_vf__' in result.kernel_source, result.kernel_source
    return result
tilelang.compile = capture

@T.prim_func
def direct_smoke(A: T.Tensor((256,), 'float32'), B: T.Tensor((256,), 'float32'), C: T.Tensor((256,), 'float32')):
    with T.Kernel(2) as bx:
        a = T.alloc_shared((128,), 'float32')
        b = T.alloc_shared((128,), 'float32')
        c = T.alloc_shared((128,), 'float32')
        T.annotate_buffer_versions({a: 1, b: 1, c: 1})
        T.copy(A[bx * 128:bx * 128 + 128], a)
        T.copy(B[bx * 128:bx * 128 + 128], b)
        with T.SimdVF():
            mask = T.simd.pset(32)
            for i in T.serial(2):
                value = T.simd.vadd(T.simd.vld(a[i * 64]), T.simd.vld(b[i * 64]), mask)
                scaled = T.simd.vmul(value, T.simd.vdup(0.25, 'float32'), mask)
                T.simd.vsts(c[i * 64], T.simd.vexp(scaled, mask), mask)
        T.copy(c, C[bx * 128:bx * 128 + 128])

if os.environ.get('A5_REDUCTION_SMOKE') != 'lowprecision':
    kernel = tilelang.compile(direct_smoke,target='ascend',execution_backend='tvm_ffi',pass_configs={'tl.enable_auto_schedule':True})
    x, y = torch.randn(256,device='npu'), torch.randn(256,device='npu')
    z = torch.empty_like(x)
    kernel(x,y,z)
    torch.testing.assert_close(z,torch.exp((x+y)*0.25),rtol=1e-4,atol=1e-5)
    torch.npu.synchronize()
    print('DIRECT_SMOKE_PASS',flush=True)

sys.path.insert(0,str(root/'pytorch/test/_inductor/tilelang'))
from test_ascend_reduction import TestAscendReduction
from torch_npu._inductor.tilelang.codegen.tilelang_scheduling import TileLangScheduling
def reject_fallback(self, node, reason):
    raise RuntimeError('TileLang fallback forbidden in validation: ' + reason)
TileLangScheduling._fallback_to_triton = reject_fallback
checks = []
original_check = TestAscendReduction.check_reduction
def record_check(self, fn, x, **kwargs):
    entry = {'test': self._testMethodName, 'shape': list(x.shape), 'dtype': str(x.dtype),
             'ops': list(fn.__code__.co_names), 'first_kernel': len(compiles), 'passed': False}
    checks.append(entry)
    try:
        source = original_check(self, fn, x, **kwargs)
        (evidence/f'wrapper-{len(checks)-1}.py').write_text(source)
        entry['passed'] = True
        return source
    except Exception:
        entry['error'] = traceback.format_exc()
        print(entry['error'], flush=True)
        raise
    finally:
        entry['compile_calls'] = len(compiles) - entry['first_kernel']
        (evidence/'cases.json').write_text(json.dumps(checks, indent=2))
        print('REDUCTION_CASE ' + json.dumps(entry), flush=True)
TestAscendReduction.check_reduction = record_check
if os.environ.get('A5_REDUCTION_SMOKE') == 'lowprecision':
    case = TestAscendReduction('test_softmax')
    suite = unittest.TestSuite()
    for dtype in (torch.float16, torch.bfloat16):
        def check(dtype=dtype):
            case.check_reduction(lambda a: torch.softmax(a, -1), torch.randn((3, 65), device='npu', dtype=dtype), atol=2e-3, rtol=2e-2)
        suite.addTest(unittest.FunctionTestCase(check, description=str(dtype)))
elif os.environ.get('A5_REDUCTION_SMOKE') == '1':
    case = TestAscendReduction('test_special_values')
    suite = unittest.TestSuite()
    for name, fn in [('sum', lambda a: a.sum(-1)), ('max', lambda a: a.amax(-1)), ('softmax', lambda a: torch.softmax(a, -1))]:
        def check(fn=fn):
            case.check_reduction(fn, torch.randn((3, 65), device='npu') - 1)
        suite.addTest(unittest.FunctionTestCase(check, description=name))
else:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestAscendReduction)
result=unittest.TextTestRunner(verbosity=2).run(suite)
report={'tests': result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'tilelang_compile_calls':len(compiles),'compute_cases':len(checks),'passed_compute_cases':sum(c['passed'] for c in checks)}
(evidence/'results.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
