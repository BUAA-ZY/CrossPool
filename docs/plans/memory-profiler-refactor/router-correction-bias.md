# Router Correction Bias: Upstream Evidence

This note distinguishes the pinned kernel's observable contract from a model's
selection-correction semantics. It records research, not an accepted execution
change. The inspected release is SGLang `v0.5.20`.

## Introduction And Stated Motivation

The original Softmax Top-K kernel was imported from vLLM in March 2025. That
import did not expose a correction-bias argument. [PR #4302](https://github.com/sgl-project/sglang/pull/4302)

`correction_bias` and `moe_softcapping` were introduced together by commit
[`f363279`](https://github.com/sgl-project/sglang/commit/f363279648f61362f2ef4fc3e447b7d9cafa793b)
on October 30, 2025. Its message is `sync router`. The enclosing
[PR #12413](https://github.com/sgl-project/sglang/pull/12413) was merged as
[`c0652d9`](https://github.com/sgl-project/sglang/commit/c0652d907b2ee39d010598ec4c5f7f287fa41947)
on October 31. Both kernel branches added bias before Softmax from the outset.

The author's PR description identifies cleanup and additional arguments, but
does not identify a model or explain the bias placement. The reviewed discussion
and review comments are Gemini bot entries, not maintainer rationale. The
changed files contain bindings and kernel changes,
but no model integration or new tests. [PR files and discussion](https://github.com/sgl-project/sglang/pull/12413/files)

## Pinned Contract

The AOT kernel computes the following transformation before Top-K:

```python
logits = gating_output.float()
if moe_softcapping:
    logits = tanh(logits / moe_softcapping) * moe_softcapping
if correction_bias is not None:
    logits = logits + correction_bias
scores = softmax(logits)
```

Both selection and returned weights use these probabilities. Optional
renormalization applies afterward. This is logit bias, not an additive correction
used only to select experts. [AOT source](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/kernels/aot/csrc/moe/moe_topk_softmax_kernels.cu)

The later JIT migration explicitly preserves the AOT device code. Its pinned
test reference independently states the same ordering and tests non-null bias.
This establishes a maintained low-level contract, but not its original model
motivation. [Merged PR #34509](https://github.com/sgl-project/sglang/pull/34509),
[reference and bias tests](https://github.com/sgl-project/sglang/blob/v0.5.20/test/registered/kernels/ops/moe/test_moe_topk_softmax.py#L58-L118)

## Model Consumers

Searching the installed release's Python consumers found no serving call to
this exact operator with non-null `correction_bias`. The high-level fallback
calls it with four arguments, leaving bias as `None`; the normal CUDA Softmax
route instead invokes `moe_fused_gate` with no bias. This is a bounded code-search
result, not proof about external consumers.
[routing dispatch](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/layers/moe/topk.py)

LongCat has Softmax selection correction, but explicitly uses the native Torch
route. That route selects using activated scores plus bias and gathers weights
from the uncorrected scores. It is therefore not evidence that LongCat uses the
AOT kernel's pre-Softmax-bias behavior.
[LongCat](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/models/longcat_flash.py#L275-L282),
[Torch reference](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/srt/layers/moe/topk.py#L750-L794)

## Conclusion

Pre-Softmax bias is explicit and tested upstream behavior. Available public
history does not establish why it was named correction bias or which concrete
serving model needs its non-null form. Treating it as compatible with a model's
selection-only correction solely because the parameter names match would be
incorrect. No runtime tests or device experiments were performed for this note.
