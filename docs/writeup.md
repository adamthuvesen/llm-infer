# What the llm-infer Experiments Found

`llm-infer` is a small inference engine for `Esme-214M-Chat`, a 214M-parameter model trained from
scratch in [`esme-pretrain`](https://github.com/adamthuvesen/esme-pretrain). It owns the whole
serving path: paged KV cache, continuous batching, batched prefill, CUDA graphs, FlashInfer
attention, and an OpenAI-compatible server. On one A100 it serves 3.1x the tokens per second of
batched HuggingFace `generate()`. vLLM on the same host is 1.9x to 3.1x faster than `llm-infer`.

This page tells the story of the 22 experiments in [findings.md](findings.md). Each entry there
has the full method and the numbers. Entries 12 to 22 cite records committed in `assets/`.
Entries 1 to 11 name raw records that stay outside the repository. Entry numbers appear below in
brackets.

## The method

Three rules decide what counts as a result.

**Reference before speed.** A tok/s number exists only after its tokens match the fp32
full-recompute oracle on the same prompts. Numerical ties are the only allowed difference
([scoping.md](scoping.md)). A fast path that changes the output is a bug.

**Same-container comparisons only.** Identical code moved about ±20% between Modal containers
[1]. So every A/B runs both arms in one container, in alternating order where possible, and a
ratio never crosses containers.

**Predictions before the run.** Entries 17 and 19 to 22 committed their predictions and decision
rule before any GPU time was spent. For the paired A/Bs (entries 19, 21, and 22), a script,
`scripts/judge_paired_ab.py`, applies the rule to the records from two containers. The statistic is the median paired change per round,
with a 95% bootstrap interval [19]. A claim needs exact output parity and an interval below zero
in both containers.

The third rule changed outcomes. It caught an underpowered pilot [19], and it rejected
speculation even though some prompts accepted two draft tokens per step [22].

## What moved the number

At 214M parameters, a decode step is about 2,100 small kernel launches [2]. The cost is the
number of kernels, not FLOPs. Most of the gains came from launching less.

- **Piecewise CUDA graphs** replay the step as a few launches. Throughput rose 177% at batch 8
  and 16% at batch 256 [3]. The ablation ladder shows graphs are the largest single lever [13].
- **`torch.compile` lost.** It cut launches by the same amount but ran at 118.7 tok/s against
  425.0 for manual graphs at batch 8 [4]. It was removed.
- **Ragged batched prefill** replaced a serial prefill loop that took 67% to 84% of the profile
  at batch 64 to 256 [5]. Batch-64 time to first token fell by 11x to 66x [6]. This closed most
  of the high-batch gap to vLLM, from 6.6–13.7x down to 1.7–2.6x [12].
- **Attention inside the graph.** Grouped graphs capture KV writes and FlashInfer attention too.
  They pay only while the batch holds its captured size [7]. Grouping all 30 layers beat the
  four-layer group by 2.9% to 4.6% per step, with exact parity, so the server now groups every
  layer [21].
- **A hidden sync.** FlashInfer's `plan()` copied page metadata to the CPU on every step [17].
  Passing host copies removed the sync and made grouped decode 1.6% to 3.8% faster [19]. The sync
  had also hidden a race on a pinned buffer. The race stalled two runs silently, and a stack dump
  from a new stall guard found it. A CUDA event now fences the buffer.

## What did not

Negative results stay in the log, because they explain the design.

- **Chunked prefill** was about 21% slower, with no stall it could fix at this size [10].
- **Operator fusion** halved the kernel count but did not make the step faster [14, 15]. The
  timeline trace meant to explain why was inconclusive, because the tracer itself changed the
  timing [16].
- **Prompt-lookup speculation** was 3.5x to 4.2x slower per request on CUDA [22]. A verify step
  runs eager, outside the decode window and its graphs. It cost about ten graphed decode steps
  and emitted two tokens. Speculation stays on only for local CPU serving.
- **A gain from arm order.** A server A/B that ran every piecewise row first showed grouped
  graphs 35% faster at sampled batch 64. With balanced order, that gain disappeared. The batch-1
  gain also depends on the host: +10.3% in one container and −1.3% in another [18]. Benchmark
  records now store the host CPU.

## The numerical contract

bf16 decode does not reproduce fp32 tokens exactly. The gate accepts a divergence only at a
near-tie, where the fp32 margin between the top two tokens is under 0.1 logits [11]. That bound
was an audited choice, not a measurement. Entry 20 measured the error on 1,024 prompts. The
statistic is the competitor shrink: how much bf16 error closes the fp32 gap between the winning
token and a competitor. Its 99.9th percentile was 0.25 logits. 7 of 807 flips had an fp32 gap
above 0.1, and no flip occurred where the gap was 0.3 or more. The bound stays at 0.1 until the
options in entry 20 are decided.

Sampled output had its own contract error. Exact sampled tokens across batch shapes are not a
valid target under bf16. The contract became same-shape seeded replay, and sampled serving became
3x to 5x faster [9].

## What is open

- The batch-1 gap to vLLM is still 3.1x [12]. Small non-GEMM kernels take 67% of a batch-1 step
  [14], so they are the next target.
- A graphed, planned multi-token verify step could change the speculation result [22].
- Should the bf16 tie bound stay at 0.1 or move to the measured 0.25 [20]?

## Check it yourself

`make reproduce` needs no GPU. It checks the committed benchmark records against the public tables,
regenerates the README figures, and reruns the judge on each registered A/B. The judge must give
the verdict that findings.md reports. The Modal commands that produce new records are listed in
each entry and in [benchmark.md](benchmark.md).
