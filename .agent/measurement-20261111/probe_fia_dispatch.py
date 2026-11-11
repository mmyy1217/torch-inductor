"""Prove which aclnn entry point torch_npu.npu_fused_infer_attention_score resolves to.

Two independent pieces of evidence:
1. Dynamic-symbol resolution: which aclnnFusedInferAttentionScoreV* symbols the
   loaded libraries export, and which one libtorch_npu actually binds.
2. Runtime dispatch: LD_DEBUG=bindings records the resolved symbol at call time.
"""

import os
import subprocess
import sys


def main():
    import torch
    import torch_npu

    lib = os.path.join(os.path.dirname(torch_npu.__file__), "lib", "libtorch_npu.so")
    print("torch_npu:", torch_npu.__version__)
    print("libtorch_npu:", lib)

    # 1. Exported symbols in the two libraries that can provide the entry point.
    for path in (lib, "/usr/local/Ascend/cann-9.3.0/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/x86_64/libopapi.so"):
        out = subprocess.run(["nm", "-D", "--defined-only", path],
                             capture_output=True, text=True).stdout
        names = sorted({
            line.split()[-1] for line in out.splitlines()
            if "aclnnFusedInferAttentionScore" in line and " T " in line
        })
        print(f"\n{os.path.basename(path)} defined aclnn entries:")
        for n in names:
            print("  ", n)

    # 2. Which of them libtorch_npu has undefined references to.
    out = subprocess.run(["nm", "-D", "--undefined-only", lib],
                         capture_output=True, text=True).stdout
    needed = sorted({
        line.split()[-1] for line in out.splitlines()
        if "aclnnFusedInferAttentionScore" in line
    })
    print("\nlibtorch_npu undefined aclnnFusedInferAttentionScore refs:")
    for n in needed:
        print("  ", n)

    # 3. Runtime binding trace around one real call.
    print("\n--- LD_DEBUG=bindings trace ---")
    script = (
        "import torch, torch_npu\n"
        "torch.npu.set_device(0)\n"
        "q=torch.randn(1,4,64,128,device='npu',dtype=torch.bfloat16)\n"
        "k=v=torch.randn(1,4,64,128,device='npu',dtype=torch.bfloat16)\n"
        "torch_npu.npu_fused_infer_attention_score(q,k,v,num_heads=4,"
        "num_key_value_heads=4,scale=0.08838834764831845,input_layout='BNSD',"
        "softmax_lse_flag=False)\n"
        "torch.npu.synchronize()\n"
    )
    env = dict(os.environ, LD_DEBUG="bindings")
    proc = subprocess.run([sys.executable, "-P", "-c", script],
                          capture_output=True, text=True, env=env)
    hits = [l for l in proc.stderr.splitlines()
            if "FusedInferAttentionScore" in l]
    for line in hits:
        print(line)
    print("exit:", proc.returncode)


if __name__ == "__main__":
    main()
