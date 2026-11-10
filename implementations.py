"""Both implementations use the original Inductor backend and lowering."""


def compile_attention(fn, implementation):
    import torch
    import torch_npu._inductor.config as config

    if implementation not in ("triton", "tilelang"):
        raise ValueError(f"Unknown implementation: {implementation}")
    if not hasattr(config.flex_attention, "implementation"):
        raise RuntimeError(
            "Install this branch's torch_npu Python sources; the wheel lacks native TileLang routing"
        )
    config.flex_attention.implementation = implementation
    return torch.compile(
        fn,
        backend="inductor",
        fullgraph=True,
        dynamic=False,
        options={
            "npu_backend": "tilelang" if implementation == "tilelang" else "default"
        },
    )
