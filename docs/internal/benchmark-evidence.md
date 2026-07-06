# Benchmark Evidence Notes

This is the maintainer evidence behind the public benchmark record. Keep raw run commands,
technique-gallery detail, and lab notes here rather than in the public README path.

## Runbook

GPU records use Modal A100 workers and require a staged Esme export bundle:

```bash
modal run scripts/modal_esme_three_way.py --command headline
modal run scripts/modal_esme_batch_curve.py --command curve
modal run scripts/modal_esme_technique_gallery.py --command gallery
uv run scripts/plot_benchmark_curve.py
```

Each harness writes raw JSON to `bench-results/` (gitignored). The curated public curve record
is committed at `assets/esme-batch-curve.json`.

## Same-Run Rule

The Esme decode loop is host-CPU-bound at 214M, and A100 containers vary in GPU SKU, SM clock,
and host CPU. Identical code has measured around +/-20% across containers. Decision-grade
comparisons therefore run back to back in one container: headline rows share a container, the
batch sweep runs in one container, and technique on/off pairs run in one container. Same-run
ratios are the useful number; cross-date raw tok/s is not.

For profile archaeology and decode-window attribution, see `docs/internal/esme-decode-overhead.md`.

## Headline Record

Run `2026-07-02`, A100-80GB, 64 concurrent chat requests, up to 256 new tokens, greedy,
1 warmup + 3 measured iterations:

| model | system | reference agreement | median s | output tok | tok/s | vs floor |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `Esme-214M-Chat` | `hf_sequential` | yes, 24 exact + 40 ties, 0 non-tie | 301.545 | 10,432 | 34.6 | 1x |
| `Esme-214M-Chat` | `llm_infer` | yes, 32 exact + 32 ties, 0 non-tie | 10.902 | 10,144 | 930.4 | 26.9x |

## Curve Evidence

The committed curve record is a separate same-container batch sweep from the headline table.
It checks both plotted systems at every batch size against the fp32 reference, with zero
non-tie divergences. At batch 256, the engine peaked at 384 used 128-token blocks: about
1.5 GB of KV at 30,720 bytes/token. A contiguous max-length layout for 256 requests at Esme's
1024-token context would reserve about 8 GB up front.

| concurrent requests | llm_infer tok/s | hf_sequential tok/s |
| ---: | ---: | ---: |
| 8 | 206.0 | 47.8 |
| 16 | 388.2 | 44.4 |
| 32 | 720.1 | 45.6 |
| 64 | 1,153.0 | 43.5 |
| 128 | 1,908.9 | 44.9 |
| 256 | 2,748.5 | 45.4 |

At 256 concurrent requests, the engine is 60.5x the same-row measured naive-HF floor.

## Technique Gallery

Each row below ran on the headline engine configuration (bf16 flash-attn) and passed the fp32
reference gate with zero non-tie divergences.

### Paged KV

The batch-256 curve row is the paged-KV evidence: requests allocate blocks lazily as sequences
grow and return blocks when they finish. The right side of the curve fits because requests do
not reserve memory they have not written.

### Prefix Caching

16 sibling requests shared one 513-token prompt, 32 new tokens each.

| prefix caching | prompt tokens prefilled | median wall | reference gate |
| --- | ---: | ---: | --- |
| on (shared group) | 513 | 1.701 s | 16/16 exact |
| off | 8,208 | 2.375 s | 16/16 exact |

End-to-end speedup: 1.40x on this workload.

### Chunked Prefill

8 short chats decoded mid-stream while 4 long prompts (711 tokens each) arrived in a burst.
The comparison used `prefill_chunk_size=128` vs whole-prompt prefill with decode window 1.

| config | max prompt tokens in one step | worst in-flight stall | long prompts served after | total wall |
| --- | ---: | ---: | ---: | ---: |
| whole-prompt prefill | 2,844 | 5.9x a decode step | 1.84 s | 6.50 s |
| chunked (128) | 512 | 6.4x a decode step | 3.40 s | 7.89 s |

The structural bound works, but at 214M the prefill pass is launch-bound, so the latency
protection is not visible on this model size and costs about 21% total wall.

### Request Preemption

12 chat requests x 96 new tokens ran in a 40-block pool where worst-case reservation needs
about 96 blocks.

| scheduler | preemptions | completed | wall | reference gate |
| --- | ---: | ---: | ---: | --- |
| preemption (recompute) | 7 | 12/12 | 8.77 s | 12/12 exact-or-tie, 0 non-tie |
| reserve (control) | 0 | 12/12 | 11.49 s | 12/12 exact-or-tie, 0 non-tie |

This row proves the pressure path: real evictions, recompute resume, and exact completions.

### Speculative Decoding

One repetition-heavy request, 128 new tokens, prompt-lookup drafts up to 4 tokens, greedy
verifier.

| speculative | median wall | tok/s | reference gate |
| --- | ---: | ---: | --- |
| on | 3.742 s | 34.2 | exact |
| off | 5.514 s | 23.2 | exact |

Speedup: 1.47x batch-1 latency, 63 verify steps emitting 2.0 tokens per verify step on
average. This path is off by default and makes no headline speed claim.
