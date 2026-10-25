# What the llm-infer Experiments Found

`llm-infer` is a small inference engine for `Esme-214M-Chat`, a 214M-parameter model trained from
scratch in [`esme-pretrain`](https://github.com/adamthuvesen/esme-pretrain). It owns the whole
serving path: paged KV cache, continuous batching, batched prefill, CUDA graphs, FlashInfer
attention, and an OpenAI-compatible server. On one A100 it serves 3.1x the tokens per second of
batched HuggingFace `generate()`. vLLM on the same host is 1.2x to 1.6x faster than `llm-infer`.

This page tells the story of the 24 experiments in [findings.md](findings.md). Each entry there
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

**Predictions before the run.** Entries 17 and 19 to 24 committed their predictions and decision
rule before any GPU time was spent. For the paired A/Bs (entries 19, 21, 22, and 23), a script,
`scripts/judge_paired_ab.py`, applies the rule to the records from two containers. The statistic
is the median paired change per round, with a 95% bootstrap interval [19]. A qualified claim needs
exact output parity and an interval below zero in both containers. Some arms cannot match token
for token by design, such as a fused kernel that rounds differently. Their gains are reported as
raw timing, and only when both arms have zero non-tie divergences from fp32 [23].

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
  four-layer group by 2.9% to 4.6% per step, with exact parity, so the server groups every layer
  [21].
- **A hidden sync.** FlashInfer's `plan()` copied page metadata to the CPU on every step [17].
  Passing host copies removed the sync and made grouped decode 1.6% to 3.8% faster [19]. Without
  the sync, a CUDA event must keep the planner's pinned buffer intact until the GPU has copied it.
- **Fused elementwise kernels.** Compiling RMSNorm and RoPE halved the kernels but did not speed
  up the step while `plan()` still synced [15, 16]. On the full-depth graphs, the same fusion cut
  37% to 44% of each step [23]. It passed the near-tie gate, added about 40 s to a cold start, and
  cut vLLM's lead in the same run from 2.6x to 1.6x at batch 1 and from 1.6x to 1.2x at batch 256
  [24]. It is the CUDA default.

## What did not

Negative results stay in the log, because they explain the design.

- **Chunked prefill** was about 21% slower, with no stall it could fix at this size [10].
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
above 0.1, and no flip occurred where the gap was 0.3 or more. The bound stays at 0.1. At 0.25 it
would sit at the ceiling of normal bf16 error and could not see a bug of that size. A decode-step
divergence with an fp32 gap of at most 0.25 and exact direct parity is a known near-tie, within
the envelope entry 20 measured (Esme chat bundle, short Dolly prompts, A100). It still gets a
written review, and the review cites entry 20. A prefill divergence, or one outside that
envelope, gets a full review.

Sampled output had its own contract error. Exact sampled tokens across batch shapes are not a
valid target under bf16. The contract became same-shape seeded replay, and sampled serving became
3x to 5x faster [9].

## What is open

- vLLM is still 1.6x faster at batch 1 on the same host [24]. The next measurement is a profile
  of the fused step, to find what makes up that gap.
- A graphed, planned multi-token verify step could change the speculation result [22].

## Check it yourself

`make reproduce` needs no GPU. It checks the committed benchmark records against the public tables,
regenerates the README figures, and reruns the judge on each registered A/B. The judge must give
the verdict that findings.md reports. The Modal commands that produce new records are listed in
each entry and in [benchmark.md](benchmark.md).
