import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { appendEventRowContent } from "./event_row.js";
import { theaterGeometry } from "./theater_layout.js";
import { buildTraceModel, parseJsonlTrace } from "./trace_loader.js";

test("loads the committed schema-v3 fixture into a renderable model", async () => {
  const text = await readFile(new URL("../docs/assets/esme_kv_trace_schema_v3.jsonl", import.meta.url), "utf8");
  const events = parseJsonlTrace(text);
  const model = buildTraceModel(events);

  assert.equal(events[0].sequence, 1);
  assert.deepEqual(
    events.map((event) => event.sequence),
    Array.from({ length: events.length }, (_, index) => index + 1),
  );
  assert.equal(model.requestList.length, 3);
  for (const request of model.requestList) {
    assert.deepEqual(
      request.decodes.flatMap((decode) => decode.tokenIds),
      request.finish.tokenIds,
    );
  }
  assert.ok(model.requestList.every((request) => request.decodes.some((decode) => decode.tokenSource === "prefill")));
  assert.ok(model.batchSignals.length > 0);
  assert.ok(model.throughputSignals.length > 0);
  assert.ok(model.pressureSamples.some((sample) => sample.reservedBlocks > 0));

  // Real block lifecycle is present and the wall can show it.
  assert.equal(model.hasBlockLifecycle, true);
  assert.ok(model.maxAllocatedBlocks > 0);
  assert.ok(model.requestList.every((request) => request.blockEvents.length > 0));
  assert.ok(model.pressureSamples.some((sample) => sample.allocatedBlocks > 0));
  assert.ok(model.pressureSamples.some((sample) => sample.allocatedBlocks === 0));
});

test("models block lifecycle without free-before-alloc and keeps the pool valid", async () => {
  const text = await readFile(new URL("../docs/assets/esme_kv_trace_schema_v3.jsonl", import.meta.url), "utf8");
  const events = parseJsonlTrace(text);

  const live = new Set();
  let poolSize = null;
  for (const event of events) {
    if (event.event !== "block_allocated" && event.event !== "block_freed") continue;
    assert.equal(event.block_count, event.block_ids.length);
    assert.ok(event.pool_used >= 0 && event.pool_free >= 0);
    poolSize ??= event.pool_used + event.pool_free;
    assert.equal(event.pool_used + event.pool_free, poolSize);
    for (const id of event.block_ids) {
      if (event.event === "block_allocated") {
        assert.ok(!live.has(id), `block ${id} allocated while live`);
        live.add(id);
      } else {
        assert.ok(live.has(id), `free-before-alloc of block ${id}`);
        live.delete(id);
      }
    }
    assert.equal(live.size, event.pool_used);
  }
  assert.equal(live.size, 0, "every block returned to the pool");
});

test("models preemption and resume on the affected request's lane", async () => {
  const text = await readFile(new URL("../docs/assets/esme_kv_trace_schema_v3.jsonl", import.meta.url), "utf8");
  const events = parseJsonlTrace(text);
  const model = buildTraceModel(events);

  const preemptEvents = events.filter((event) => event.event === "request_preempted");
  const resumeEvents = events.filter((event) => event.event === "request_resumed");
  assert.ok(preemptEvents.length > 0, "fixture must contain a preemption");
  assert.ok(resumeEvents.length > 0, "fixture must contain a resume");

  // The preempted request's lane carries the eviction and the later resume.
  const preemptedId = preemptEvents[0].request_id;
  const lane = model.requestList.find((request) => request.requestId === preemptedId);
  assert.ok(lane.preempts.length > 0, "preempted lane must record the eviction");
  assert.ok(lane.resumes.length > 0, "preempted lane must record the resume");
  assert.equal(lane.preempts[0].reason, "kv_pressure");
  assert.ok(lane.preempts[0].freedBlocks >= 1);

  // It still finishes with all its tokens — preemption never drops output.
  assert.deepEqual(
    lane.decodes.flatMap((decode) => decode.tokenIds),
    lane.finish.tokenIds,
  );
});

test("validates the request_preempted reason", () => {
  const bad = JSON.stringify({
    event: "request_preempted",
    schema_version: 3,
    sequence: 1,
    step: 0,
    request_id: "x",
    preempt_reason: "swapped-to-disk",
  });
  assert.throws(() => parseJsonlTrace(bad), /unknown preempt_reason/);
});

test("rejects a trace with a sequence gap (playback assumes contiguous ids)", () => {
  const text = [
    {
      event: "request_admitted",
      schema_version: 3,
      sequence: 1,
      step: 0,
      request_id: "r",
      prompt_tokens: 2,
      max_new_tokens: 4,
      reserved_blocks: 1,
    },
    { event: "decode_step", schema_version: 3, sequence: 3, step: 1, request_ids: ["r"], token_ids: [9] },
  ]
    .map((event) => JSON.stringify(event))
    .join("\n");
  assert.throws(() => parseJsonlTrace(text), /Sequence gap/);
});

test("rejects a trace whose sequence ids do not start at one", () => {
  const text = [
    {
      event: "request_admitted",
      schema_version: 3,
      sequence: 10,
      step: 0,
      request_id: "r",
      prompt_tokens: 2,
      max_new_tokens: 4,
      reserved_blocks: 1,
    },
    { event: "decode_step", schema_version: 3, sequence: 11, step: 1, request_ids: ["r"], token_ids: [9] },
  ]
    .map((event) => JSON.stringify(event))
    .join("\n");
  assert.throws(() => parseJsonlTrace(text), /start at 1/);
});

test("rejects a decode_step whose token_ids are not integers", () => {
  const text = JSON.stringify({
    event: "decode_step",
    schema_version: 3,
    sequence: 1,
    step: 0,
    request_ids: ["r"],
    token_ids: ["nope"],
  });
  assert.throws(() => parseJsonlTrace(text), /token_ids must be an array of integers/);
});

test("rejects a multi-request decode_step that is not one token per request", () => {
  const text = JSON.stringify({
    event: "decode_step",
    schema_version: 3,
    sequence: 1,
    step: 0,
    request_ids: ["a", "b"],
    token_ids: [9],
  });
  assert.throws(() => parseJsonlTrace(text), /one token_id per request/);
});

test("theater layout keeps the timeline and lanes inside the rendered frame", () => {
  const layout = theaterGeometry(4);

  const textSafetyGap = 12;
  assert.ok(layout.frameLeft < layout.left);
  assert.ok(layout.left < layout.right);
  assert.ok(layout.right < layout.frameRight);
  assert.ok(layout.gridTop - 8 + textSafetyGap < layout.laneTop);
  assert.equal(layout.width, 1000);
  assert.equal(layout.laneHeight, 88);
  assert.ok(layout.gridTop < layout.laneTop);
  assert.ok(layout.gridBottom <= layout.height);
});

test("rejects malformed throughput samples before inspector rendering", () => {
  const text = JSON.stringify({
    event: "tokens_per_second_sampled",
    schema_version: 3,
    sequence: 1,
    step: 0,
    tokens_per_second: "fast",
    tokens_emitted: 1,
    total_generated_tokens: 1,
    elapsed_seconds: 0.1,
  });
  assert.throws(() => parseJsonlTrace(text), /tokens_per_second_sampled/);
});

test("keeps speculative-style multi-token decode bursts on one request lane", () => {
  const text = [
    {
      event: "request_admitted",
      schema_version: 3,
      sequence: 1,
      step: 0,
      request_id: "spec",
      prompt_tokens: 3,
      max_new_tokens: 5,
      reserved_blocks: 2,
    },
    {
      event: "decode_step",
      schema_version: 3,
      sequence: 2,
      step: 1,
      request_ids: ["spec"],
      token_ids: [41, 42, 43],
      tokens_emitted: 3,
      batch_size: 1,
    },
  ]
    .map((event) => JSON.stringify(event))
    .join("\n");

  const model = buildTraceModel(parseJsonlTrace(text));
  assert.deepEqual(model.requestList[0].decodes[0].tokenIds, [41, 42, 43]);
});

test("re-admission after preemption preserves the original lane and generated-token count", () => {
  const text = [
    {
      event: "request_admitted",
      schema_version: 3,
      sequence: 1,
      step: 0,
      request_id: "r",
      prompt_tokens: 3,
      max_new_tokens: 5,
      reserved_blocks: 2,
    },
    {
      event: "decode_step",
      schema_version: 3,
      sequence: 2,
      step: 1,
      request_ids: ["r"],
      token_ids: [41],
      tokens_emitted: 1,
      batch_size: 1,
    },
    {
      event: "request_preempted",
      schema_version: 3,
      sequence: 3,
      step: 2,
      request_id: "r",
      preempt_reason: "kv_pressure",
      block_count: 1,
      generated_tokens: 1,
      pool_used: 0,
      pool_free: 4,
    },
    {
      event: "request_admitted",
      schema_version: 3,
      sequence: 4,
      step: 3,
      request_id: "r",
      prompt_tokens: 3,
      max_new_tokens: 5,
      reserved_blocks: 2,
    },
    {
      event: "request_resumed",
      schema_version: 3,
      sequence: 5,
      step: 3,
      request_id: "r",
      prompt_tokens: 3,
      generated_tokens: 1,
      cached_tokens: 3,
      pool_used: 1,
      pool_free: 3,
    },
  ]
    .map((event) => JSON.stringify(event))
    .join("\n");

  const model = buildTraceModel(parseJsonlTrace(text));
  assert.equal(model.requestList.length, 1);
  assert.equal(model.requestList[0].firstSequence, 1);
  assert.equal(model.requestList[0].resumes[0].generatedTokens, 1);
  assert.equal(model.pressureSamples.at(-1).logicalTokens, 4);
});

test("renders hostile event labels as text, not markup", () => {
  const writes = [];
  const documentRef = {
    createElement(tagName) {
      const node = {
        tagName,
        children: [],
        className: "",
        _textContent: "",
        append(...children) {
          this.children.push(...children);
        },
        set innerHTML(_value) {
          throw new Error("innerHTML must not be used for trace-derived labels");
        },
        get textContent() {
          return this._textContent;
        },
        set textContent(value) {
          writes.push(String(value));
          this._textContent = String(value);
        },
      };
      return node;
    },
  };
  const row = documentRef.createElement("button");
  const hostile = 'decode step · <img src=x onerror="globalThis.pwned=true">';

  appendEventRowContent(documentRef, row, { sequence: 7, step: 3 }, hostile);

  assert.equal(row.children.length, 3);
  assert.equal(row.children[1].tagName, "strong");
  assert.equal(row.children[1].textContent, hostile);
  assert.ok(writes.includes(hostile));
});

test("rejects non-string request labels before rendering", () => {
  const text = JSON.stringify({
    event: "request_admitted",
    schema_version: 3,
    sequence: 1,
    step: 0,
    request_id: { html: "<script>alert(1)</script>" },
  });

  assert.throws(() => parseJsonlTrace(text), /request_id must be a string/);
});

test("rejects event-specific required fields before model building", () => {
  const missingRequestId = JSON.stringify({
    event: "request_admitted",
    schema_version: 3,
    sequence: 1,
    step: 0,
    prompt_tokens: 3,
    max_new_tokens: 5,
    reserved_blocks: 2,
  });
  assert.throws(() => parseJsonlTrace(missingRequestId), /request_admitted must have a non-empty request_id/);

  const missingFinishedTokens = JSON.stringify({
    event: "request_finished",
    schema_version: 3,
    sequence: 1,
    step: 0,
    request_id: "r",
    generated_tokens: 1,
    reason: "length",
  });
  assert.throws(() => parseJsonlTrace(missingFinishedTokens), /request_finished token_ids/);
});
