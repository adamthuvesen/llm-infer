import { appendEventRowContent } from "./event_row.js";
import { buildTraceModel, eventLabel, parseJsonlTrace } from "./trace_loader.js";

const SAMPLE_TRACE_PATH = "../docs/assets/kv_trace_schema_v2.jsonl";
const NS = "http://www.w3.org/2000/svg";

const state = {
  events: [],
  model: null,
  cursor: 1,
  playing: false,
  timer: null,
  stepMs: 650,
};

const els = {
  titleMeta: document.querySelector("#titleMeta"),
  fileInput: document.querySelector("#fileInput"),
  playButton: document.querySelector("#playButton"),
  speedSelect: document.querySelector("#speedSelect"),
  scrubber: document.querySelector("#scrubber"),
  scrubberValue: document.querySelector("#scrubberValue"),
  status: document.querySelector("#status"),
  theaterSvg: document.querySelector("#theaterSvg"),
  eventList: document.querySelector("#eventList"),
  inspectorTitle: document.querySelector("#inspectorTitle"),
  inspectorMeta: document.querySelector("#inspectorMeta"),
  inspectorJson: document.querySelector("#inspectorJson"),
  eventAnatomy: document.querySelector("#eventAnatomy"),
  pressureLogical: document.querySelector("#pressureLogical"),
  pressureReserved: document.querySelector("#pressureReserved"),
  pressureActive: document.querySelector("#pressureActive"),
  pressureLogicalFill: document.querySelector("#pressureLogicalFill"),
  pressureReservedFill: document.querySelector("#pressureReservedFill"),
  currentPhase: document.querySelector("#currentPhase"),
  currentStep: document.querySelector("#currentStep"),
  currentBatch: document.querySelector("#currentBatch"),
  currentThroughput: document.querySelector("#currentThroughput"),
};

init();

async function init() {
  bindControls();
  try {
    const response = await fetch(SAMPLE_TRACE_PATH);
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    await loadTraceText(await response.text(), SAMPLE_TRACE_PATH);
  } catch (error) {
    setStatus(`Open a trace JSONL file to begin. Sample fetch failed: ${error.message}`, true);
  }
}

function bindControls() {
  els.fileInput.addEventListener("change", async (event) => {
    const [file] = event.target.files;
    if (!file) {
      return;
    }
    try {
      await loadTraceText(await file.text(), file.name);
    } catch (error) {
      pause();
      setStatus(`Could not load ${file.name}: ${error.message}`, true);
    }
  });

  els.playButton.addEventListener("click", () => {
    state.playing ? pause() : play();
  });

  els.speedSelect.addEventListener("change", () => {
    state.stepMs = Number(els.speedSelect.value);
    if (state.playing) {
      pause();
      play();
    }
  });

  els.scrubber.addEventListener("input", () => {
    pause();
    state.cursor = Number(els.scrubber.value);
    render();
  });
}

async function loadTraceText(text, sourceName) {
  const events = parseJsonlTrace(text);
  state.events = events;
  state.model = buildTraceModel(events);
  state.cursor = events[0].sequence;
  els.scrubber.min = events[0].sequence;
  els.scrubber.max = state.model.maxSequence;
  els.scrubber.value = state.cursor;
  els.titleMeta.textContent = `${displaySourceName(sourceName)} · ${events.length} events · ${state.model.requestList.length} reqs`;
  setStatus("Trace loaded", false);
  render();
}

function play() {
  if (!state.model) {
    return;
  }
  state.playing = true;
  els.playButton.textContent = "Pause";
  state.timer = window.setInterval(() => {
    if (state.cursor >= state.model.maxSequence) {
      pause();
      return;
    }
    state.cursor += 1;
    render();
  }, state.stepMs);
}

function pause() {
  state.playing = false;
  els.playButton.textContent = "Play";
  if (state.timer !== null) {
    window.clearInterval(state.timer);
    state.timer = null;
  }
}

function render() {
  if (!state.model) {
    return;
  }
  els.scrubber.value = state.cursor;
  els.scrubberValue.textContent = `event ${state.cursor}/${state.model.maxSequence}`;
  renderTheater();
  renderEventList();
  renderInspector();
  renderPressure();
  renderCurrentLoop();
}

function renderTheater() {
  const width = 1180;
  const height = 650;
  const left = 180;
  const timelineRight = 850;
  const laneTop = 112;
  const laneHeight = 78;
  const signalTop = 486;
  const current = currentEvent();
  const pressure = currentPressure();
  const cursorX = xForStep(current.step, left, timelineRight);
  resetSvg(els.theaterSvg, width, height);
  defs(els.theaterSvg);

  rect(els.theaterSvg, 18, 20, width - 36, height - 40, "stage-backdrop");
  text(els.theaterSvg, 34, 54, "KV CACHE THEATER", "svg-kicker");
  text(els.theaterSvg, 34, 82, eventHeadline(current), "svg-headline");
  text(els.theaterSvg, 34, 106, eventEffect(current), "svg-subhead");
  schedulerLoop(els.theaterSvg, 925, 66, current);
  memoryWall(els.theaterSvg, 910, 150, pressure);

  for (let step = 0; step <= state.model.maxStep; step += 1) {
    const x = xForStep(step, left, timelineRight);
    line(els.theaterSvg, x, laneTop - 34, x, signalTop + 122, "stage-grid");
    text(els.theaterSvg, x, laneTop - 45, String(step), "tick-label", "middle");
  }

  line(els.theaterSvg, cursorX, laneTop - 54, cursorX, signalTop + 128, "cursor-line");
  circle(els.theaterSvg, cursorX, laneTop - 24, 8, "cursor-pulse");
  text(els.theaterSvg, cursorX + 14, laneTop - 18, "now", "cursor-label");

  state.model.requestList.forEach((request, index) => {
    renderRequestLane(request, index, { left, right: timelineRight, top: laneTop, laneHeight });
  });
  renderSignalRibbons({ left, right: timelineRight, top: signalTop });
  renderEventComets(current, { left, right: timelineRight, top: 438 });
}

function schedulerLoop(svg, cx, cy, event) {
  circle(svg, cx, cy, 38, "loop-ring");
  circle(svg, cx, cy, 19, event.event === "batch_size_changed" ? "loop-core hot" : "loop-core");
  text(svg, cx, cy - 54, "scheduler loop", "wall-label", "middle");
  text(svg, cx, cy + 5, `s${event.step}`, "loop-step", "middle");
  const phases = ["admit", "prefill", "decode", "finish"];
  phases.forEach((phase, index) => {
    const angle = -Math.PI / 2 + index * (Math.PI / 2);
    const x = cx + Math.cos(angle) * 58;
    const y = cy + Math.sin(angle) * 58;
    circle(svg, x, y, 5, event.event.includes(phase) ? "phase-dot active" : "phase-dot");
    text(svg, x, y + 18, phase, "phase-label", "middle");
  });
}

function memoryWall(svg, x, y, pressure) {
  const blockCount = state.model.maxReservedBlocks;
  const cols = 3;
  const cell = 34;
  const gap = 8;
  const rows = Math.ceil(blockCount / cols);
  text(svg, x, y - 28, "reserved KV blocks", "wall-label");
  text(svg, x + 188, y - 28, `${pressure.reservedBlocks}/${blockCount}`, "wall-value", "end");
  for (let index = 0; index < blockCount; index += 1) {
    const col = index % cols;
    const row = Math.floor(index / cols);
    rect(
      svg,
      x + col * (cell + gap),
      y + row * (cell + gap),
      cell,
      cell,
      index < pressure.reservedBlocks ? "cache-block active" : "cache-block",
    );
  }
  const railY = y + rows * (cell + gap) + 34;
  text(svg, x, railY, "logical cache footprint", "wall-label");
  rect(svg, x, railY + 16, 188, 12, "memory-rail");
  rect(
    svg,
    x,
    railY + 16,
    Math.max(4, (pressure.logicalTokens / state.model.maxLogicalTokens) * 188),
    12,
    "memory-fill",
  );
  text(svg, x, railY + 50, `${pressure.logicalTokens} cached/generated token slots`, "wall-note");
}

function renderRequestLane(request, index, frame) {
  const y = frame.top + index * frame.laneHeight;
  const labelY = y + 28;
  text(els.theaterSvg, 34, labelY, request.requestId, "lane-title");
  text(els.theaterSvg, 34, labelY + 22, laneMeta(request), "lane-subtitle");
  line(els.theaterSvg, frame.left, y + 34, frame.right, y + 34, "lane-track");

  for (const chunk of request.chunks) {
    const x = xForStep(chunk.step, frame.left, frame.right);
    const visible = chunk.sequence <= state.cursor;
    const chunkWidth = Math.max(42, ((chunk.endPos - chunk.startPos) / chunk.totalPromptTokens) * 118);
    rect(els.theaterSvg, x - 6, y + 12, chunkWidth, 44, visible ? "prefill" : "future");
    text(
      els.theaterSvg,
      x + chunkWidth / 2 - 6,
      y + 39,
      `${chunk.startPos}-${chunk.endPos}`,
      visible ? "mark-text" : "mark-text muted",
      "middle",
    );
  }

  for (const decode of request.decodes) {
    const x = xForStep(decode.step, frame.left, frame.right);
    const visible = decode.sequence <= state.cursor;
    const tokenText =
      decode.tokenIds.length > 1 ? `${decode.tokenIds.length} tok` : `${decode.tokenIds[0] ?? ""}`;
    const className = tokenClass(decode, visible);
    circle(els.theaterSvg, x, y + 64, 14, className);
    text(els.theaterSvg, x, y + 69, tokenText, visible ? "token-text" : "token-text muted", "middle");
  }

  if (request.finish) {
    const x = xForStep(request.finish.step, frame.left, frame.right);
    const visible = request.finish.sequence <= state.cursor;
    circle(els.theaterSvg, x + 42, y + 34, 7, visible ? "finish" : "future-dot");
    text(els.theaterSvg, x + 55, y + 39, request.finish.reason, visible ? "finish-text" : "finish-text muted");
  }
}

function renderSignalRibbons(frame) {
  text(els.theaterSvg, 34, frame.top + 9, "loop signals", "svg-kicker");
  signalAxis(frame.left, frame.right, frame.top + 32, "batch");
  signalAxis(frame.left, frame.right, frame.top + 82, "tok/s");
  signalAxis(frame.left, frame.right, frame.top + 132, "reserved");

  const visibleBatch = state.model.batchSignals.filter((signal) => signal.sequence <= state.cursor);
  const maxBatch = Math.max(...state.model.batchSignals.map((signal) => signal.batchSize + signal.waiting), 1);
  drawPolyline(
    visibleBatch.map((signal) => [
      xForStep(signal.step, frame.left, frame.right),
      yInBand(signal.batchSize, maxBatch, frame.top + 16, frame.top + 48),
    ]),
    "batch-line",
  );

  const visibleThroughput = state.model.throughputSignals.filter((signal) => signal.sequence <= state.cursor);
  const maxTps = Math.max(...state.model.throughputSignals.map((signal) => signal.tokensPerSecond), 1);
  drawPolyline(
    visibleThroughput.map((signal) => [
      xForStep(signal.step, frame.left, frame.right),
      yInBand(signal.tokensPerSecond, maxTps, frame.top + 66, frame.top + 98),
    ]),
    "throughput-line",
  );

  const visiblePressure = state.model.pressureSamples.filter((sample) => sample.sequence <= state.cursor);
  drawPolyline(
    visiblePressure.map((sample) => [
      xForStep(sample.step, frame.left, frame.right),
      yInBand(sample.reservedBlocks, state.model.maxReservedBlocks, frame.top + 116, frame.top + 148),
    ]),
    "pressure-line",
  );
}

function renderEventComets(current, frame) {
  const recent = state.model.events
    .filter((event) => event.sequence <= state.cursor)
    .slice(-10);
  recent.forEach((event, index) => {
    const x = xForStep(event.step, frame.left, frame.right);
    const r = event.sequence === current.sequence ? 7 : 4;
    circle(els.theaterSvg, x, frame.top + index * 3, r, event.sequence === current.sequence ? "event-comet active" : "event-comet");
  });
}

function signalAxis(left, right, y, label) {
  text(els.theaterSvg, 100, y + 4, label, "signal-label", "end");
  line(els.theaterSvg, left, y, right, y, "signal-axis");
}

function renderEventList() {
  const fragment = document.createDocumentFragment();
  for (const event of state.model.events) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = event.sequence === state.cursor ? "event-row is-current" : "event-row";
    if (event.sequence > state.cursor) {
      button.classList.add("is-future");
    }
    appendEventRowContent(document, button, event, eventLabel(event));
    button.addEventListener("click", () => {
      pause();
      state.cursor = event.sequence;
      render();
    });
    fragment.append(button);
  }
  els.eventList.replaceChildren(fragment);
}

function renderInspector() {
  const event = currentEvent();
  els.inspectorTitle.textContent = eventLabel(event);
  els.inspectorMeta.textContent = `sequence ${event.sequence} · step ${event.step}`;
  els.inspectorJson.textContent = JSON.stringify(event, null, 2);
  renderAnatomy(event);
}

function renderAnatomy(event) {
  const items = eventAnatomy(event);
  const fragment = document.createDocumentFragment();
  for (const item of items) {
    const row = document.createElement("div");
    row.className = "anatomy-row";
    const label = document.createElement("span");
    label.textContent = item.label;
    const value = document.createElement("strong");
    value.textContent = item.value;
    row.append(label, value);
    fragment.append(row);
  }
  els.eventAnatomy.replaceChildren(fragment);
}

function eventAnatomy(event) {
  if (event.event === "request_admitted") {
    return [
      { label: "Request", value: event.request_id },
      { label: "Prompt budget", value: `${event.prompt_tokens} prompt · ${event.max_new_tokens} new` },
      { label: "Reservation", value: `${event.reserved_blocks} KV blocks committed` },
      { label: "Why it matters", value: "Scheduler admits only if the full cache budget fits." },
    ];
  }
  if (event.event === "prefill_chunk_progress") {
    return [
      { label: "Request", value: event.request_id },
      { label: "Chunk cached", value: `${event.start_pos}-${event.end_pos} of ${event.total_prompt_tokens}` },
      { label: "Cache effect", value: `${event.cached_tokens} prompt positions are now logical cache state` },
      { label: "Why it matters", value: event.completed ? "Prefill can now sample the first token." : "Decode waits while the prompt cache fills." },
    ];
  }
  if (event.event === "decode_step") {
    return [
      { label: "Requests", value: event.request_ids.join(", ") },
      { label: "Token source", value: event.token_source ?? "decode" },
      { label: "Tokens", value: event.token_ids.join(", ") },
      { label: "Why it matters", value: "Every generated token appears here before final output." },
    ];
  }
  if (event.event === "batch_size_changed") {
    return [
      { label: "Running", value: `${event.previous_batch_size} → ${event.batch_size}` },
      { label: "Waiting", value: String(event.waiting) },
      { label: "Why it matters", value: "Continuous batching changes occupancy at loop boundaries." },
    ];
  }
  if (event.event === "request_finished") {
    return [
      { label: "Request", value: event.request_id },
      { label: "Reason", value: event.reason },
      { label: "Final tokens", value: event.token_ids.join(", ") },
      { label: "Why it matters", value: "Finished requests release scheduler budget after this point." },
    ];
  }
  if (event.event === "tokens_per_second_sampled") {
    return [
      { label: "Step emitted", value: `${event.tokens_emitted} tokens` },
      { label: "Total", value: `${event.total_generated_tokens} tokens` },
      { label: "Throughput", value: `${event.tokens_per_second.toFixed(1)} tok/s` },
      { label: "Why it matters", value: "Throughput is sampled from real emitted tokens." },
    ];
  }
  return [
    { label: "Event", value: event.event },
    { label: "Why it matters", value: "Trace event in engine emission order." },
  ];
}

function renderPressure() {
  const pressure = currentPressure();
  const logicalPercent = (pressure.logicalTokens / state.model.maxLogicalTokens) * 100;
  const reservedPercent = (pressure.reservedBlocks / state.model.maxReservedBlocks) * 100;
  els.pressureLogical.textContent = `${pressure.logicalTokens} logical tokens`;
  els.pressureReserved.textContent = `${pressure.reservedBlocks} reserved blocks`;
  els.pressureActive.textContent = `${pressure.activeRequests} active requests`;
  els.pressureLogicalFill.style.width = `${Math.max(3, logicalPercent)}%`;
  els.pressureReservedFill.style.width = `${Math.max(3, reservedPercent)}%`;
}

function renderCurrentLoop() {
  const event = currentEvent();
  const pressure = currentPressure();
  const batch = currentBatchSignal();
  const throughput = currentThroughputSignal();
  els.currentPhase.textContent = event.event.replaceAll("_", " ");
  els.currentStep.textContent = `step ${event.step}`;
  els.currentBatch.textContent = `${batch.batchSize} running · ${batch.waiting} waiting`;
  els.currentThroughput.textContent =
    throughput === null ? "no sample yet" : `${throughput.tokensPerSecond.toFixed(1)} tok/s`;
  els.currentPhase.dataset.event = event.event;
  els.currentBatch.dataset.active = String(pressure.activeRequests);
}

function currentEvent() {
  return state.model.events.find((event) => event.sequence === state.cursor) ?? state.model.events[0];
}

function currentPressure() {
  const samples = state.model.pressureSamples.filter((sample) => sample.sequence <= state.cursor);
  return samples.at(-1) ?? state.model.pressureSamples[0];
}

function currentBatchSignal() {
  const signals = state.model.batchSignals.filter((signal) => signal.sequence <= state.cursor);
  return signals.at(-1) ?? { batchSize: 0, waiting: 0 };
}

function currentThroughputSignal() {
  const signals = state.model.throughputSignals.filter((signal) => signal.sequence <= state.cursor);
  return signals.at(-1) ?? null;
}

function eventHeadline(event) {
  if (event.event === "decode_step") {
    return `${event.token_source ?? "decode"} token emission`;
  }
  return event.event.replaceAll("_", " ");
}

function eventEffect(event) {
  if (event.event === "decode_step") {
    return `${event.request_ids.join(", ")} emits ${event.token_ids.join(", ")}`;
  }
  if (event.request_id) {
    return event.request_id;
  }
  if (event.event === "batch_size_changed") {
    return `${event.previous_batch_size} → ${event.batch_size} running, ${event.waiting} waiting`;
  }
  return "engine event in sequence order";
}

function laneMeta(request) {
  const parts = [`${request.promptTokens ?? "?"}p`, `${request.maxNewTokens ?? "?"}g`, `${request.reservedBlocks ?? 0} blocks`];
  if (request.prefixGroupId) {
    parts.push(`group ${request.prefixGroupId}`);
  }
  return parts.join(" · ");
}

function tokenClass(decode, visible) {
  if (!visible) {
    return "token future-token";
  }
  if (decode.tokenSource === "prefill") {
    return "token prefill-token";
  }
  if (decode.tokenSource === "speculative") {
    return "token speculative-token";
  }
  return "token decode-token";
}

function xForStep(step, left, right) {
  const span = Math.max(state.model.maxStep, 1);
  return left + (step / span) * (right - left);
}

function yInBand(value, maxValue, top, bottom) {
  const ratio = maxValue === 0 ? 0 : value / maxValue;
  return bottom - ratio * (bottom - top);
}

function resetSvg(svg, width, height) {
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.replaceChildren();
}

function defs(svg) {
  const defsNode = document.createElementNS(NS, "defs");
  defsNode.innerHTML = `
    <linearGradient id="stageGlow" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#123e3a"/>
      <stop offset="48%" stop-color="#121820"/>
      <stop offset="100%" stop-color="#241a30"/>
    </linearGradient>
    <linearGradient id="memoryFill" x1="0" x2="1">
      <stop offset="0%" stop-color="#55a7ff"/>
      <stop offset="100%" stop-color="#31c7a8"/>
    </linearGradient>
  `;
  svg.append(defsNode);
}

function line(svg, x1, y1, x2, y2, className) {
  const node = document.createElementNS(NS, "line");
  node.setAttribute("x1", x1);
  node.setAttribute("y1", y1);
  node.setAttribute("x2", x2);
  node.setAttribute("y2", y2);
  node.setAttribute("class", className);
  svg.append(node);
}

function rect(svg, x, y, width, height, className) {
  const node = document.createElementNS(NS, "rect");
  node.setAttribute("x", x);
  node.setAttribute("y", y);
  node.setAttribute("width", width);
  node.setAttribute("height", height);
  node.setAttribute("rx", 8);
  node.setAttribute("class", className);
  svg.append(node);
}

function circle(svg, cx, cy, r, className) {
  const node = document.createElementNS(NS, "circle");
  node.setAttribute("cx", cx);
  node.setAttribute("cy", cy);
  node.setAttribute("r", r);
  node.setAttribute("class", className);
  svg.append(node);
}

function text(svg, x, y, value, className, anchor = "start") {
  const node = document.createElementNS(NS, "text");
  node.setAttribute("x", x);
  node.setAttribute("y", y);
  node.setAttribute("class", className);
  node.setAttribute("text-anchor", anchor);
  node.textContent = value;
  svg.append(node);
}

function drawPolyline(points, className) {
  if (points.length < 1) {
    return;
  }
  const node = document.createElementNS(NS, "polyline");
  node.setAttribute("points", points.map(([x, y]) => `${x},${y}`).join(" "));
  node.setAttribute("class", className);
  els.theaterSvg.append(node);
}

function setStatus(message, isError) {
  els.status.textContent = message;
  els.status.classList.toggle("is-error", isError);
}

function displaySourceName(sourceName) {
  return sourceName.split(/[\\/]/).at(-1) || sourceName;
}
