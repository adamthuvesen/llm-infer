# Reference-Policy Retro Audit

This is the durable Phase 3 inventory for benchmark evidence produced before policy v2. Raw
performance records under `bench-results/` are gitignored and remain immutable. The table below
is the record: it is self-contained, closed (every row ends in "no rerun"), and is not expected
to change. Smoke runs, duplicate timestamps, JSONL resume logs, and diagnostic attachments are
deliberately not all catalogued.

## Dispositions

| Claim | Policy | Reference status | Direct parity | Headline eligible | Disposition | Rerun decision |
| --- | ---: | --- | --- | --- | --- | --- |
| Public FlashInfer batch curve | v1 | Qualified; zero non-tie divergences | Not applicable; both systems were checked against fp32 | Yes | Qualified | No rerun |
| Paged-KV capacity evidence | v1 | Qualified through the batch-256 curve row | Not applicable | Yes | Qualified | No rerun |
| Phase 0 engine matrix | v1 | Passing rows plus one documented context-768 non-tie | Not applicable | No, as one family | Historical review | Preserve raw timing; no rerun |
| Persistent HTTP greedy baseline | v1 | Recorded greedy requests pass | Not applicable | No | Qualified diagnostic | No rerun |
| Persistent HTTP sampled baseline | v1 | Old cross-shape exact-token gate is no longer valid | Not applicable | No | Historical review | Replace under Phase 5; no retro rerun |
| Current vLLM comparison | v1 | Every recorded row passes fp32 | Not applicable | No public headline | Qualified | No rerun |
| Piecewise decode capture profile | v1 | Zero recorded non-ties | Not recorded between configurations | No | Qualified diagnostic | No rerun |
| Stable native-page metadata A/B | v1 | Batch 1/64 qualified; batch 8 needs numerical review | Shared continuation documented; raw outputs absent | No | Qualified with a limited row | **No rerun; do not rebuild the removed legacy comparator** |
| Packed-prefill A/B and divergence review | v1 | Qualified after documented numerical review | Stored outputs and parity allow offline backfill | No | Offline-backfillable | No GPU rerun |
| Mixed-load prefill/decode A/B | v1 | Qualified after documented numerical review | Stored mode outputs allow offline backfill | No | Offline-backfillable | No GPU rerun |
| Fixed-buffer FlashInfer graph probe | v1 | Kernel-level reference not applicable | Exact graph/ordinary comparisons across page boundaries | No | Qualified diagnostic | No rerun |
| Two-layer grouped graph | v1 | Batch 1 qualified; batch 8 needs numerical review | Not stored directly | No | Superseded by four layers | No rerun |
| Four-layer grouped graph | v1 timing + v2 parity | Batch 1 qualified; batch 8 shares one numerical review | Exact at batch 8 | No absolute headline; relative A/B qualified | Qualified relative result | No further rerun |
| Historical prefix-cache gallery | v1 | Qualified on the then-current flash-attn backend | Not stored directly | No | Historical review | No retro rerun |
| Historical chunked-prefill gallery | v1 | Qualified on the then-current flash-attn backend | Not stored directly | No | Historical review | No retro rerun |
| Historical preemption gallery | v1 | Qualified on the then-current flash-attn backend | Not stored directly | No | Historical review | No retro rerun |
| Historical speculative-decode gallery | v1 | Exact on one repetition-heavy request | Exact for that request | No | Historical review | No retro rerun |
| Smoke, failed, and duplicate intermediate runs | v1 | Not audited for current claims | Not applicable | No | Superseded | No rerun or reinterpretation |

## Records and limitations

- The public curve is durable in `assets/esme-batch-curve.json`; its local raw record is optional.
- The Phase 0 matrix keeps raw timing for the known failing context-768 row, but that row is not
  correctness evidence and remains ineligible for a speed claim.
- The page-plan batch-8 baseline and candidate produced the same continuation and shared the same
  fp32 margin outside the old automatic boundary. That supports the relative wall result, but the
  absent raw outputs prevent retroactively claiming a stronger direct-parity record.
- Packed prefill and mixed load stored both modes' outputs. Their direct parity can be recomputed
  offline without pretending every historical iteration retained output tokens.
- The wrapper probe proves fixed-address replay across the tested page boundaries only. It is not
  end-to-end model evidence.
- The four-layer batch-8 correctness-only follow-up recorded exact baseline/candidate parity. Both
  choose token 13204 at `esme-007` step 22 while fp32 chooses 1616 at a 0.13044-logit margin. This
  qualifies the old same-run relative A/B result without turning it into an absolute headline.
  The performance matrix was not rerun.
- Historical gallery rows used the former flash-attn default. They remain dated technique evidence,
  not current-backend headline claims.

No historical result file is rewritten by this audit. Policy v2 applies to new records; policy-v1
claims above keep their original measurements, explicit limits, and rerun decisions.
