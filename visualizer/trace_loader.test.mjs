import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { appendEventRowContent } from "./event_row.js";
import { buildTheaterLayout, validateTheaterLayout } from "./theater_layout.js";
import { buildTraceModel, parseJsonlTrace } from "./trace_loader.js";

test("loads the committed schema-v3 fixture into a renderable model", async () => {
  const text = await readFile(new URL("../docs/assets/kv_trace_schema_v3.jsonl", import.meta.url), "utf8");
  const events = parseJsonlTrace(text);
  const model = buildTraceModel(events);

  assert.equal(events[0].sequence, 1);
  assert.deepEqual(
    events.map((event) => event.sequence),
    Array.from({ length: events.length }, (_, index) => index + 1),
  );
  assert.equal(model.requestList.length, 6);
  assert.ok(model.requestList.some((request) => request.chunks.length >= 3));
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

test("models block lifecycle honestly: no free-before-alloc, pool stays valid", async () => {
  const text = await readFile(new URL("../docs/assets/kv_trace_schema_v3.jsonl", import.meta.url), "utf8");
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

test("theater layout keeps headline, cursor marker, ticks, and lanes separated", () => {
  const layout = buildTheaterLayout(4);

  assert.equal(validateTheaterLayout(layout), true);
  assert.ok(layout.header.subheadY < layout.cursorCy - 8);
  assert.ok(layout.cursorLabelY < layout.laneTop);
  assert.ok(layout.tickY < layout.laneTop);
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
