import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { appendEventRowContent } from "./event_row.js";
import { buildTraceModel, parseJsonlTrace } from "./trace_loader.js";

test("loads the committed schema-v2 fixture into a renderable model", async () => {
  const text = await readFile(new URL("../docs/assets/kv_trace_schema_v2.jsonl", import.meta.url), "utf8");
  const events = parseJsonlTrace(text);
  const model = buildTraceModel(events);

  assert.equal(events[0].sequence, 1);
  assert.deepEqual(
    events.map((event) => event.sequence),
    Array.from({ length: events.length }, (_, index) => index + 1),
  );
  assert.equal(model.requestList.length, 4);
  assert.ok(model.requestList.some((request) => request.chunks.length >= 3));
  assert.ok(model.batchSignals.length > 0);
  assert.ok(model.throughputSignals.length > 0);
  assert.ok(model.pressureSamples.some((sample) => sample.reservedBlocks > 0));
});

test("keeps speculative-style multi-token decode bursts on one request lane", () => {
  const text = [
    {
      event: "request_admitted",
      schema_version: 2,
      sequence: 1,
      step: 0,
      request_id: "spec",
      prompt_tokens: 3,
      max_new_tokens: 5,
      reserved_blocks: 2,
    },
    {
      event: "decode_step",
      schema_version: 2,
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
    schema_version: 2,
    sequence: 1,
    step: 0,
    request_id: { html: "<script>alert(1)</script>" },
  });

  assert.throws(() => parseJsonlTrace(text), /request_id must be a string/);
});
