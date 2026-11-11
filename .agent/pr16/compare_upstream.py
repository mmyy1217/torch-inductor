"""Direct preallocated TileLang comparison; native eager scope stays separate."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

p = argparse.ArgumentParser()
p.add_argument('--impl', choices=['upstream32', 'adapted32', 'adapted28', 'native'], required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
import torch
import torch_npu
import tilelang

torch.manual_seed(2026)
torch.npu.set_device(0)
# Identical logical BNSD inputs. Flatten H,M without a copy for upstream MQA.
q, k, v = [torch.randn(s, dtype=torch.bfloat16).to('npu') for s in
           [(1,32,8192,128),(1,1,8192,128),(1,1,8192,128)]]
scale = 128 ** -0.5
out = torch.full_like(q, float('nan'))
root = Path(__file__).resolve().parent
meta = dict(impl=a.impl, shape=[1,32,1,8192,8192,128], scale=scale,
            torch=torch.__version__, torch_npu=torch_npu.__version__,
            tilelang=str(Path(tilelang.__file__).resolve()), samples=30,
            scope='operator_eager_with_allocation' if a.impl=='native' else 'kernel_direct_preallocated',
            note='No Inductor. Head-major flattened upstream input; layout conversion excluded. Not original FIA V5 reproduction.')
if a.impl == 'native':
    def operation():
        return torch_npu.npu_fusion_attention(q,k,v,32,'BNSD',scale=scale,keep_prob=1.0,sparse_mode=0)[0]
else:
    path = root / ('upstream_core.py' if a.impl=='upstream32' else a.impl+'.py')
    spec = importlib.util.spec_from_file_location('candidate', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    meta['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    if a.impl == 'upstream32':
        prim = module.flash_attention_fwd(32*8192,8192,128,num_blocks=32,softmax_scale=scale)
        inputs = (q.view(-1,128), k.view(-1,128), v.view(-1,128), out.view(-1,128))
    else:
        prim = module.dense_attention_prim_func((1,32,8192,8192,128,1),scale)
        inputs = (q,k,v,out)
    kernel = tilelang.compile(prim,target='ascend',out_idx=None)
    (a.out/'device_code.txt').write_text(kernel.get_kernel_source())
    (a.out/'prim_func.py').write_text(prim.script())
    def operation():
        kernel(*inputs)
        return out
(a.out/'metadata.json').write_text(json.dumps(meta,indent=2))
# Independent FP32 oracle for deterministic query rows across every head.
rows = torch.tensor([0,1,127,128,4095,8191],device='npu')
expected = torch.softmax(q[:,:,rows,:].float() @ k.float().transpose(-2,-1)*scale,dim=-1) @ v.float()
def check(stage):
    actual = operation()
    torch.npu.synchronize()
    if not bool(torch.isfinite(actual).all()):
        raise RuntimeError(stage+': nonfinite output')
    selected = actual[:,:,rows,:].float()
    torch.testing.assert_close(selected,expected,atol=.03,rtol=.03)
    (a.out/(stage+'.json')).write_text(json.dumps({'finite':True,'sampled_rows_per_head':6,'max_abs_error':float((selected-expected).abs().max())}))
with torch.no_grad():
    check('pre')
    for _ in range(10): operation()
    torch.npu.synchronize()
    values=[]
    for _ in range(30):
        start,end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
        start.record()
        result=operation()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end)*1000)
    check('post')
result=dict(meta, samples_us=values, median_us=statistics.median(values))
(a.out/'result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
