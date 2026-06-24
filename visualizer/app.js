import { appendEventRowContent } from "./event_row.js";
import { buildTraceModel, eventLabel, parseJsonlTrace } from "./trace_loader.js";

const SAMPLE_TRACE_PATH = "../docs/assets/kv_trace_schema_v2.jsonl";
const NS = "http://www.w3.org/2000/svg";
const REDUCE = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

const state = {
  events: [],
  model: null,
  cursor: 1,
  prevCursor: 0,
  prevReserved: 0,
  playing: false,
  timer: null,
  stepMs: 650,
  theater: null,
};

const els = {
  traceMeta: q("#traceMeta"),
  fileInput: q("#fileInput"),
  playButton: q("#playButton"),
  playLabel: q("#playLabel"),
  speedSeg: q("#speedSeg"),
  scrubber: q("#scrubber"),
  scrubFill: q("#scrubFill"),
  scrubHead: q("#scrubHead"),
  seqNow: q("#seqNow"),
  seqTotal: q("#seqTotal"),
  phaseChip: q("#phaseChip"),
  phaseName: q("#phaseName"),
  status: q("#status"),
  gStep: q("#gStep"),
  gStepFoot: q("#gStepFoot"),
  gRunning: q("#gRunning"),
  gWaiting: q("#gWaiting"),
  occBar: q("#occBar"),
  gTps: q("#gTps"),
  tpsNote: q("#tpsNote"),
  gGen: q("#gGen"),
  sparkTps: q("#sparkTps"),
  sparkGen: q("#sparkGen"),
  theater: q("#theater"),
  kvGrid: q("#kvGrid"),
  kvReserved: q("#kvReserved"),
  kvTotal: q("#kvTotal"),
  kvLogical: q("#kvLogical"),
  kvLogicalFill: q("#kvLogicalFill"),
  nowTitle: q("#nowTitle"),
  nowMeta: q("#nowMeta"),
  anatomy: q("#anatomy"),
  rawJson: q("#rawJson"),
  eventList: q("#eventList"),
  tooltip: q("#tooltip"),
};

init();

async function init() {
  bindControls();
  setupTooltip();
  try {
    const response = await fetch(SAMPLE_TRACE_PATH);
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    await loadTraceText(await response.text(), SAMPLE_TRACE_PATH, true);
  } catch (error) {
    setStatus(`load a JSONL trace to begin · ${error.message}`, true);
  }
}

function bindControls() {
  els.fileInput.addEventListener("change", async (event) => {
    const [file] = event.target.files;
    if (!file) return;
    try {
      await loadTraceText(await file.text(), file.name, false);
    } catch (error) {
      pause();
      setStatus(`could not load ${file.name} · ${error.message}`, true);
    }
  });

  els.playButton.addEventListener("click", () => (state.playing ? pause() : play()));

  els.speedSeg.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-ms]");
    if (!button) return;
    [...els.speedSeg.children].forEach((child) => child.classList.toggle("is-active", child === button));
    state.stepMs = Number(button.dataset.ms);
    if (state.playing) {
      pause();
      play();
    }
  });

  els.scrubber.addEventListener("input", () => {
    pause();
    setCursor(Number(els.scrubber.value));
  });
}

async function loadTraceText(text, sourceName, isSample = false) {
  const events = parseJsonlTrace(text);
  state.events = events;
  state.model = buildTraceModel(events);
  state.cursor = events[0].sequence;
  state.prevCursor = 0;
  state.prevReserved = 0;
  state.theater = null;

  els.scrubber.min = events[0].sequence;
  els.scrubber.max = state.model.maxSequence;
  els.scrubber.value = state.cursor;
  els.seqTotal.textContent = pad(state.model.maxSequence);
  els.traceMeta.textContent = `${baseName(sourceName)} · ${events.length} events · ${state.model.requestList.length} requests · ${state.model.maxStep + 1} steps${isSample ? " · synthetic sample" : ""}`;
  els.tpsNote.textContent = isSample ? "simulated step clock" : "";

  buildKvWall();
  setStatus("trace loaded", false);
  render();

  if (!REDUCE) {
    els.theater.classList.remove("intro");
    void els.theater.offsetWidth;
    els.theater.classList.add("intro");
  }
}

function play() {
  if (!state.model) return;
  if (state.cursor >= state.model.maxSequence) setCursor(state.model.events[0].sequence);
  state.playing = true;
  els.playLabel.textContent = "Pause";
  document.body.classList.add("is-playing");
  state.timer = window.setInterval(() => {
    if (state.cursor >= state.model.maxSequence) {
      pause();
      return;
    }
    setCursor(state.cursor + 1);
  }, state.stepMs);
}

function pause() {
  state.playing = false;
  els.playLabel.textContent = "Play";
  document.body.classList.remove("is-playing");
  if (state.timer !== null) {
    window.clearInterval(state.timer);
    state.timer = null;
  }
}

function setCursor(value) {
  state.prevCursor = state.cursor;
  state.cursor = value;
  render();
}

/* ============================ render ============================ */

function render() {
  if (!state.model) return;
  const event = currentEvent();
  els.scrubber.value = state.cursor;
  els.seqNow.textContent = pad(state.cursor);

  const progress = (state.cursor - 1) / Math.max(1, state.model.maxSequence - 1);
  els.scrubFill.style.width = `${progress * 100}%`;
  els.scrubHead.style.left = `${progress * 100}%`;

  if (els.phaseChip.dataset.event !== event.event && !REDUCE) {
    els.phaseChip.animate([{ opacity: 0.3 }, { opacity: 1 }], { duration: 240, easing: "ease-out" });
  }
  els.phaseChip.dataset.event = event.event;
  els.phaseName.textContent = event.event.replaceAll("_", " ");

  renderGauges(event);
  renderTheater();
  renderKvWall();
  renderInspector(event);
  renderEventList();
}

function renderGauges(event) {
  const pressure = currentPressure();
  const batch = currentBatchSignal();
  const tps = currentThroughputSignal();

  animateNumber(els.gStep, event.step, 0);
  els.gStepFoot.textContent = `${pressure.activeRequests} active · ${state.model.maxStep + 1} total`;
  animateNumber(els.gRunning, batch.batchSize, 0);
  animateNumber(els.gWaiting, batch.waiting, 0);
  els.occBar.style.setProperty("--occ", `${(batch.batchSize / maxBatchSize()) * 100}%`);

  if (tps) {
    animateNumber(els.gTps, tps.tokensPerSecond, 1);
  } else {
    els.gTps.textContent = "—";
    delete els.gTps.dataset.v;
  }
  animateNumber(els.gGen, tps ? tps.totalGeneratedTokens : 0, 0);

  const visibleTps = state.model.throughputSignals.filter((s) => s.sequence <= state.cursor);
  const maxTps = Math.max(...state.model.throughputSignals.map((s) => s.tokensPerSecond), 1);
  drawSpark(els.sparkTps, visibleTps.map((s) => s.tokensPerSecond), maxTps, state.model.throughputSignals.length);

  const maxGen = Math.max(...state.model.throughputSignals.map((s) => s.totalGeneratedTokens), 1);
  drawSpark(els.sparkGen, visibleTps.map((s) => s.totalGeneratedTokens), maxGen, state.model.throughputSignals.length);
}

function renderTheater() {
  const t = ensureTheater();
  clear(t.scene);

  const lanes = state.model.requestList;
  const cursorX = xForStep(currentEvent().step, t.geo);

  // step grid + ticks
  const curStep = currentEvent().step;
  for (let step = 0; step <= state.model.maxStep; step += 1) {
    const x = xForStep(step, t.geo);
    const on = step === curStep;
    line(t.scene, x, t.geo.gridTop, x, t.geo.gridBottom, `grid-line${on ? " active" : ""}`);
    text(t.scene, x, t.geo.gridTop - 8, String(step), `tick-label${on ? " active" : ""}`, "middle");
  }
  text(t.scene, t.geo.left - 14, t.geo.gridTop - 8, "STEP", "axis-cap", "end");

  lanes.forEach((request, index) => renderLane(t.scene, request, index, t.geo));

  // playhead (persistent group, smoothly translated)
  t.playhead.setAttribute("transform", `translate(${cursorX}, 0)`);
  t.playLine.setAttribute("y2", t.geo.gridBottom);
  t.playGlow.setAttribute("y2", t.geo.gridBottom);
}

function renderLane(scene, request, index, geo) {
  const y0 = geo.laneTop + index * geo.laneH;
  const trackY = y0 + geo.laneH * 0.62;
  const lane = group(scene, "lane");

  rect(lane, geo.frameLeft, y0, geo.frameRight - geo.frameLeft, geo.laneH, `lane-band${index % 2 ? " alt" : ""}`, 0);

  text(lane, 16, y0 + 26, request.requestId, "lane-title");
  text(lane, 16, y0 + 44, laneMeta(request), "lane-sub");
  if (request.prefixGroupId) text(lane, 16, y0 + 60, `▦ ${request.prefixGroupId}`, "lane-group");

  // track: live up to finish, faded after — only once the request has actually finished
  const finished = request.finish && request.finish.sequence <= state.cursor;
  const finishX = finished ? xForStep(request.finish.step, geo) : geo.right;
  line(lane, geo.left, trackY, Math.max(geo.left, finishX), trackY, "lane-track");
  if (finishX < geo.right - 1) line(lane, finishX, trackY, geo.right, trackY, "lane-track done");

  // prefill chunks
  for (const chunk of request.chunks) {
    const x = xForStep(chunk.step, geo);
    const visible = chunk.sequence <= state.cursor;
    const frac = chunk.totalPromptTokens ? (chunk.endPos - chunk.startPos) / chunk.totalPromptTokens : 0.3;
    const w = Math.max(26, Math.min(74, frac * 90 + 26));
    const cls = !visible ? "chunk future" : chunk.completed ? "chunk" : "chunk partial";
    const node = rect(lane, x - w / 2, trackY - 36, w, 20, cls + enterClass(chunk.sequence, "enter-chunk"), 6);
    node.setAttribute("data-tip", `prefill · cached ${chunk.startPos}–${chunk.endPos} of ${chunk.totalPromptTokens}${chunk.completed ? " · complete" : " · filling"}`);
    text(lane, x, trackY - 22, `${chunk.startPos}–${chunk.endPos}`, `chunk-label${visible ? "" : " future"}`, "middle");
  }

  // thread the visible decode tokens so the generation sequence reads as a line
  const threadXs = request.decodes
    .filter((decode) => decode.sequence <= state.cursor)
    .map((decode) => xForStep(decode.step, geo))
    .sort((a, b) => a - b);
  if (threadXs.length >= 2) {
    const thread = document.createElementNS(NS, "polyline");
    thread.setAttribute("points", threadXs.map((x) => `${x},${trackY}`).join(" "));
    thread.setAttribute("class", "lane-thread");
    lane.append(thread);
  }

  // decode tokens on the track
  for (const decode of request.decodes) {
    const x = xForStep(decode.step, geo);
    drawToken(lane, x, trackY, decode, decode.sequence <= state.cursor);
  }

  // finish marker
  if (request.finish) {
    const x = xForStep(request.finish.step, geo);
    const visible = request.finish.sequence <= state.cursor;
    const fx = x + 26;
    const g = group(lane, visible ? enterClass(request.finish.sequence, "enter-finish") : "");
    const ring = circle(g, fx, trackY, 11, `finish-ring${visible ? "" : " future"}`);
    ring.setAttribute("data-tip", `finished · ${request.finish.reason} · ${request.finish.tokenIds.length} tokens`);
    diamond(g, fx, trackY, 6, `finish-flag${visible ? "" : " future"}`);
    text(g, fx + 16, trackY + 4, request.finish.reason, `finish-label${visible ? "" : " future"}`);
  }
}

function drawToken(scene, x, y, decode, visible) {
  const source = decode.tokenSource;
  const colour = source === "prefill" ? "tok-prefill" : source === "speculative" ? "tok-speculative" : "tok-decode";
  const enter = visible ? enterClass(decode.sequence, "enter-tok") : "";

  if (decode.tokenIds.length > 1) {
    const w = 44;
    const node = rect(scene, x - w / 2, y - 12, w, 24, `burst ${visible ? colour : "future"}${enter}`, 8);
    node.setAttribute("data-tip", `${source} burst · step ${decode.step} · tokens ${decode.tokenIds.join(", ")}`);
    text(scene, x, y + 4, `×${decode.tokenIds.length}`, `burst-id${visible ? "" : " future"}`, "middle");
    text(scene, x, y + 24, decode.tokenIds.join(" "), `tok-id${visible ? "" : " future"}`, "middle");
    return;
  }
  const node = circle(scene, x, y, 8, `tok-dot ${visible ? colour : "future"}${enter}`);
  node.setAttribute("data-tip", `${source} token · id ${decode.tokenIds[0] ?? "—"} · step ${decode.step}`);
  text(scene, x, y + 22, String(decode.tokenIds[0] ?? ""), `tok-id${visible ? "" : " future"}`, "middle");
}

function renderKvWall() {
  const pressure = currentPressure();
  const blocks = els.kvGrid.children;
  for (let i = 0; i < blocks.length; i += 1) {
    const on = i < pressure.reservedBlocks;
    blocks[i].classList.toggle("on", on);
    blocks[i].classList.toggle("just", on && i >= state.prevReserved && pressure.reservedBlocks > state.prevReserved);
  }
  state.prevReserved = pressure.reservedBlocks;

  animateNumber(els.kvReserved, pressure.reservedBlocks, 0);
  els.kvTotal.textContent = state.model.maxReservedBlocks;
  animateNumber(els.kvLogical, pressure.logicalTokens, 0, " tokens");
  els.kvLogicalFill.style.width = `${Math.max(3, (pressure.logicalTokens / state.model.maxLogicalTokens) * 100)}%`;
}

function renderInspector(event) {
  els.nowTitle.textContent = event.event.replaceAll("_", " ");
  els.nowMeta.textContent = `seq ${event.sequence} · step ${event.step}`;
  els.rawJson.textContent = JSON.stringify(event, null, 2);

  const fragment = document.createDocumentFragment();
  for (const item of eventAnatomy(event)) {
    const row = document.createElement("div");
    row.className = "row";
    const dt = document.createElement("dt");
    dt.textContent = item.label;
    const dd = document.createElement("dd");
    dd.className = item.why ? "why" : item.mono ? "mono" : "";
    dd.textContent = item.value;
    row.append(dt, dd);
    fragment.append(row);
  }
  els.anatomy.replaceChildren(fragment);
  if (!REDUCE) {
    els.anatomy.animate(
      [{ opacity: 0, transform: "translateY(6px)" }, { opacity: 1, transform: "none" }],
      { duration: 240, easing: "cubic-bezier(0.22,0.61,0.36,1)" },
    );
    els.nowTitle.animate([{ opacity: 0.35 }, { opacity: 1 }], { duration: 220, easing: "ease-out" });
  }
}

function renderEventList() {
  const fragment = document.createDocumentFragment();
  for (const event of state.model.events) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "event-row";
    button.dataset.fam = eventFamily(event);
    if (event.sequence === state.cursor) button.classList.add("is-current");
    if (event.sequence > state.cursor) button.classList.add("is-future");
    appendEventRowContent(document, button, event, eventLabel(event));
    button.addEventListener("click", () => {
      pause();
      setCursor(event.sequence);
    });
    fragment.append(button);
  }
  els.eventList.replaceChildren(fragment);
  const current = els.eventList.querySelector(".is-current");
  if (current) current.scrollIntoView({ block: "nearest" });
}

/* ============================ theater scaffolding ============================ */

function ensureTheater() {
  const laneCount = state.model.requestList.length;
  if (state.theater && state.theater.laneCount === laneCount) return state.theater;

  const geo = theaterGeometry(laneCount);
  const svg = els.theater;
  svg.setAttribute("viewBox", `0 0 ${geo.width} ${geo.height}`);
  svg.replaceChildren();

  const scene = group(svg, "");
  const playhead = group(svg, "");
  const playGlow = line(playhead, 0, geo.gridTop - 16, 0, geo.gridBottom, "playhead-glow");
  const playLine = line(playhead, 0, geo.gridTop - 16, 0, geo.gridBottom, "playhead-line");
  circle(playhead, 0, geo.gridTop - 16, 4, "playhead-cap");
  text(playhead, 8, geo.gridTop - 12, "NOW", "playhead-label", "start");

  state.theater = { laneCount, geo, scene, playhead, playLine, playGlow };
  return state.theater;
}

function theaterGeometry(laneCount) {
  const width = 1000;
  const left = 196;
  const right = width - 40;
  const top = 70;
  const laneH = 88;
  const laneTop = top;
  const height = laneTop + laneCount * laneH + 24;
  return {
    width, height, left, right,
    frameLeft: 8, frameRight: width - 8,
    laneTop, laneH,
    gridTop: top - 8,
    gridBottom: laneTop + laneCount * laneH + 4,
  };
}

/* ============================ data helpers ============================ */

function currentEvent() {
  return state.model.events.find((event) => event.sequence === state.cursor) ?? state.model.events[0];
}
function currentPressure() {
  const samples = state.model.pressureSamples.filter((s) => s.sequence <= state.cursor);
  return samples.at(-1) ?? state.model.pressureSamples[0] ?? { reservedBlocks: 0, logicalTokens: 0, activeRequests: 0 };
}
function currentBatchSignal() {
  const signals = state.model.batchSignals.filter((s) => s.sequence <= state.cursor);
  return signals.at(-1) ?? { batchSize: 0, waiting: 0 };
}
function currentThroughputSignal() {
  const signals = state.model.throughputSignals.filter((s) => s.sequence <= state.cursor);
  return signals.at(-1) ?? null;
}
function maxBatchSize() {
  return Math.max(...state.model.batchSignals.map((s) => s.batchSize), 1);
}

function eventFamily(event) {
  switch (event.event) {
    case "request_admitted": case "batch_size_changed": return "admit";
    case "prefill_chunk_started": case "prefill_chunk_progress": return "prefill";
    case "decode_step": return event.token_source === "speculative" ? "spec" : "decode";
    case "request_finished": return "finish";
    default: return "";
  }
}

function laneMeta(request) {
  return `${request.promptTokens ?? "?"}p · ${request.maxNewTokens ?? "?"}g · ${request.reservedBlocks ?? 0} blk`;
}

function eventAnatomy(event) {
  if (event.event === "request_admitted") {
    return [
      { label: "request", value: event.request_id, mono: true },
      { label: "budget", value: `${event.prompt_tokens} prompt · ${event.max_new_tokens} new`, mono: true },
      { label: "reserved", value: `${event.reserved_blocks} KV blocks`, mono: true },
      { label: "why", value: "The scheduler admits a request only when its full KV budget fits.", why: true },
    ];
  }
  if (event.event === "prefill_chunk_started") {
    return [
      { label: "request", value: event.request_id, mono: true },
      { label: "chunk", value: `${event.start_pos}–${event.end_pos} of ${event.total_prompt_tokens}`, mono: true },
      { label: "why", value: "Prefill walks the prompt in chunks so long prompts share the loop.", why: true },
    ];
  }
  if (event.event === "prefill_chunk_progress") {
    return [
      { label: "request", value: event.request_id, mono: true },
      { label: "cached", value: `${event.cached_tokens} / ${event.total_prompt_tokens} positions`, mono: true },
      { label: "complete", value: event.completed ? "yes" : "not yet", mono: true },
      { label: "why", value: event.completed ? "Prompt cache is full — decode can sample the first token." : "Decode waits while the prompt cache fills.", why: true },
    ];
  }
  if (event.event === "decode_step") {
    return [
      { label: "requests", value: event.request_ids.join(", "), mono: true },
      { label: "source", value: event.token_source ?? "decode", mono: true },
      { label: "tokens", value: event.token_ids.join(", "), mono: true },
      { label: "why", value: "Every generated token is emitted here, one loop step at a time.", why: true },
    ];
  }
  if (event.event === "batch_size_changed") {
    return [
      { label: "running", value: `${event.previous_batch_size} → ${event.batch_size}`, mono: true },
      { label: "waiting", value: String(event.waiting), mono: true },
      { label: "why", value: "Continuous batching changes occupancy at loop boundaries.", why: true },
    ];
  }
  if (event.event === "request_finished") {
    return [
      { label: "request", value: event.request_id, mono: true },
      { label: "reason", value: event.reason, mono: true },
      { label: "output", value: event.token_ids.join(", "), mono: true },
      { label: "why", value: "Finishing frees the request's reserved KV blocks for the next admit.", why: true },
    ];
  }
  if (event.event === "tokens_per_second_sampled") {
    return [
      { label: "rate", value: `${event.tokens_per_second.toFixed(1)} tok/s`, mono: true },
      { label: "step", value: `${event.tokens_emitted} emitted`, mono: true },
      { label: "total", value: `${event.total_generated_tokens} tokens · ${event.elapsed_seconds}s`, mono: true },
      { label: "why", value: "Throughput is sampled from real emitted tokens, not estimated.", why: true },
    ];
  }
  return [{ label: "event", value: event.event, mono: true }];
}

function buildKvWall() {
  const fragment = document.createDocumentFragment();
  for (let i = 0; i < state.model.maxReservedBlocks; i += 1) {
    const block = document.createElement("div");
    block.className = "kv-block";
    fragment.append(block);
  }
  els.kvGrid.replaceChildren(fragment);
}

/* ============================ svg + misc utils ============================ */

function enterClass(sequence, klass) {
  return sequence > state.prevCursor && sequence <= state.cursor ? ` ${klass}` : "";
}

function drawSpark(svg, values, maxValue, total) {
  svg.replaceChildren();
  if (values.length === 0) return;
  const W = 200, H = 44, pad = 4;
  const span = Math.max(total - 1, 1);
  const points = values.map((v, i) => {
    const x = pad + (i / span) * (W - pad * 2);
    const y = H - pad - (v / maxValue) * (H - pad * 2);
    return [x, y];
  });
  if (points.length === 1) points.unshift([pad, points[0][1]]);
  const linePts = points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const last = points.at(-1);

  const area = document.createElementNS(NS, "polygon");
  area.setAttribute("class", "area");
  area.setAttribute("points", `${pad},${H - pad} ${linePts} ${last[0].toFixed(1)},${H - pad}`);
  svg.append(area);

  const lineNode = document.createElementNS(NS, "polyline");
  lineNode.setAttribute("class", "line");
  lineNode.setAttribute("points", linePts);
  svg.append(lineNode);

  circle(svg, last[0], last[1], 2.6, "dot");
}

function group(parent, className) {
  const node = document.createElementNS(NS, "g");
  if (className) node.setAttribute("class", className.trim());
  parent.append(node);
  return node;
}
function line(parent, x1, y1, x2, y2, className) {
  const node = document.createElementNS(NS, "line");
  node.setAttribute("x1", x1); node.setAttribute("y1", y1);
  node.setAttribute("x2", x2); node.setAttribute("y2", y2);
  node.setAttribute("class", className);
  parent.append(node);
  return node;
}
function rect(parent, x, y, w, h, className, r = 8) {
  const node = document.createElementNS(NS, "rect");
  node.setAttribute("x", x); node.setAttribute("y", y);
  node.setAttribute("width", Math.max(0, w)); node.setAttribute("height", Math.max(0, h));
  node.setAttribute("rx", r);
  node.setAttribute("class", className.trim());
  parent.append(node);
  return node;
}
function circle(parent, cx, cy, r, className) {
  const node = document.createElementNS(NS, "circle");
  node.setAttribute("cx", cx); node.setAttribute("cy", cy); node.setAttribute("r", r);
  node.setAttribute("class", className.trim());
  parent.append(node);
  return node;
}
function diamond(parent, cx, cy, r, className) {
  const node = document.createElementNS(NS, "polygon");
  node.setAttribute("points", `${cx},${cy - r} ${cx + r},${cy} ${cx},${cy + r} ${cx - r},${cy}`);
  node.setAttribute("class", className.trim());
  parent.append(node);
  return node;
}
function text(parent, x, y, value, className, anchor = "start") {
  const node = document.createElementNS(NS, "text");
  node.setAttribute("x", x); node.setAttribute("y", y);
  node.setAttribute("class", className);
  node.setAttribute("text-anchor", anchor);
  node.textContent = value;
  parent.append(node);
  return node;
}
function clear(node) { node.replaceChildren(); }

function xForStep(step, geo) {
  const span = Math.max(state.model.maxStep, 1);
  return geo.left + (step / span) * (geo.right - geo.left);
}

function animateNumber(el, target, decimals = 0, suffix = "") {
  const fmt = (n) => `${decimals ? n.toFixed(decimals) : Math.round(n)}${suffix}`;
  const from = el.dataset.v !== undefined ? Number(el.dataset.v) : target;
  el.dataset.v = String(target);
  if (REDUCE || from === target) {
    el.textContent = fmt(target);
    return;
  }
  if (el._raf) cancelAnimationFrame(el._raf);
  const duration = 360;
  const start = performance.now();
  const tick = (now) => {
    const p = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = fmt(from + (target - from) * eased);
    el._raf = p < 1 ? requestAnimationFrame(tick) : null;
  };
  el._raf = requestAnimationFrame(tick);
}

function setupTooltip() {
  const tip = els.tooltip;
  const svg = els.theater;
  svg.addEventListener("mouseover", (event) => {
    const text = event.target?.getAttribute?.("data-tip");
    if (!text) return;
    tip.textContent = text;
    tip.classList.add("show");
  });
  svg.addEventListener("mouseout", (event) => {
    if (event.target?.getAttribute?.("data-tip")) tip.classList.remove("show");
  });
  svg.addEventListener("mousemove", (event) => {
    if (!tip.classList.contains("show")) return;
    const gap = 14;
    let x = event.clientX + gap;
    if (x + tip.offsetWidth > window.innerWidth - 8) x = event.clientX - gap - tip.offsetWidth;
    tip.style.left = `${x}px`;
    tip.style.top = `${event.clientY + gap}px`;
  });
}

function pad(value) { return String(value).padStart(2, "0"); }
function baseName(name) { return name.split(/[\\/]/).at(-1) || name; }
function setStatus(message, isError) {
  els.status.textContent = message;
  els.status.classList.toggle("is-error", isError);
}
function q(selector) { return document.querySelector(selector); }
