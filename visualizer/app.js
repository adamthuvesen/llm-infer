import { buildTraceModel, eventLabel, parseJsonlTrace } from "./trace_loader.js";
import { appendEventRowContent } from "./event_row.js";

const SAMPLE_TRACE_PATH = "../docs/assets/kv_trace_schema_v2.jsonl";
const NS = "http://www.w3.org/2000/svg";
const COLORS = {
  prefill: "#31c7a8",
  decode: "#f2b84b",
  finish: "#ff6f61",
  reserved: "#8e7dff",
  pressure: "#55a7ff",
  muted: "#5f6673",
  axis: "#343946",
  text: "#e9edf2",
};

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
  laneSvg: document.querySelector("#laneSvg"),
  signalSvg: document.querySelector("#signalSvg"),
  eventList: document.querySelector("#eventList"),
  inspectorTitle: document.querySelector("#inspectorTitle"),
  inspectorMeta: document.querySelector("#inspectorMeta"),
  inspectorJson: document.querySelector("#inspectorJson"),
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
  renderLanes();
  renderSignals();
  renderEventList();
  renderInspector();
  renderPressure();
  renderCurrentLoop();
}

function renderLanes() {
  const { requestList, maxStep } = state.model;
  const width = Math.max(1040, 280 + Math.max(maxStep, 1) * 86);
  const laneHeight = 76;
  const top = 40;
  const left = 170;
  const rightPad = 120;
  const height = top + requestList.length * laneHeight + 36;
  resetSvg(els.laneSvg, width, height);

  for (let step = 0; step <= maxStep; step += 1) {
    const x = xForStep(step, maxStep, left, width, rightPad);
    line(els.laneSvg, x, top - 16, x, height - 24, "grid");
    text(els.laneSvg, x, 20, String(step), "tick-label", "middle");
  }

  const current = currentEvent();
  const cursorX = xForStep(current.step, maxStep, left, width, rightPad);
  line(els.laneSvg, cursorX, top - 22, cursorX, height - 18, "cursor-line");

  requestList.forEach((request, index) => {
    const y = top + index * laneHeight;
    laneLabel(request, y);
    line(els.laneSvg, left, y + 26, width - 28, y + 26, "lane-line");

    for (const chunk of request.chunks) {
      const x = xForStep(chunk.step, maxStep, left, width, rightPad);
      const visible = chunk.sequence <= state.cursor;
      const chunkWidth = Math.max(42, ((chunk.endPos - chunk.startPos) / chunk.totalPromptTokens) * 96);
      rect(els.laneSvg, x - 5, y + 8, chunkWidth, 36, visible ? "prefill" : "future");
      text(
        els.laneSvg,
        x + chunkWidth / 2 - 5,
        y + 31,
        `${chunk.startPos}-${chunk.endPos}`,
        visible ? "mark-text" : "mark-text muted",
        "middle",
      );
    }

    for (const decode of request.decodes) {
      const x = xForStep(decode.step, maxStep, left, width, rightPad);
      const visible = decode.sequence <= state.cursor;
      const tokenText = decode.tokenIds.length > 1 ? `${decode.tokenIds.length} tok` : `${decode.tokenIds[0] ?? ""}`;
      rect(els.laneSvg, x - 12, y + 49, Math.max(34, tokenText.length * 9 + 16), 22, visible ? "decode" : "future");
      text(
        els.laneSvg,
        x + Math.max(34, tokenText.length * 9 + 16) / 2 - 12,
        y + 65,
        tokenText,
        visible ? "mark-text dark" : "mark-text muted",
        "middle",
      );
    }

    if (request.finish) {
      const x = xForStep(request.finish.step, maxStep, left, width, rightPad);
      const visible = request.finish.sequence <= state.cursor;
      circle(els.laneSvg, x + 44, y + 26, 8, visible ? "finish" : "future-dot");
      text(els.laneSvg, x + 60, y + 31, request.finish.reason, visible ? "finish-text" : "finish-text muted");
    }
  });
}

function laneLabel(request, y) {
  text(els.laneSvg, 18, y + 20, request.requestId, "lane-title");
  const parts = [
    `${request.promptTokens ?? "?"}p`,
    `${request.maxNewTokens ?? "?"}g`,
    `${request.reservedBlocks ?? 0} reserved`,
  ];
  if (request.prefixGroupId) {
    parts.push(`group ${request.prefixGroupId}`);
  }
  text(els.laneSvg, 18, y + 42, parts.join(" · "), "lane-subtitle");
}

function renderSignals() {
  const { batchSignals, throughputSignals, pressureSamples, maxStep } = state.model;
  const width = Math.max(920, 220 + Math.max(maxStep, 1) * 82);
  const height = 230;
  const left = 72;
  const rightPad = 42;
  resetSvg(els.signalSvg, width, height);
  chartFrame(width, 34, 92, "batch / waiting");
  chartFrame(width, 132, 190, "throughput / pressure");

  const visibleBatch = batchSignals.filter((signal) => signal.sequence <= state.cursor);
  const maxBatch = Math.max(...batchSignals.map((signal) => signal.batchSize + signal.waiting), 1);
  drawPolyline(
    visibleBatch.map((signal) => [
      xForStep(signal.step, maxStep, left, width, rightPad),
      yInBand(signal.batchSize, maxBatch, 34, 92),
    ]),
    "batch-line",
  );
  drawPolyline(
    visibleBatch.map((signal) => [
      xForStep(signal.step, maxStep, left, width, rightPad),
      yInBand(signal.waiting, maxBatch, 34, 92),
    ]),
    "waiting-line",
  );

  const visibleThroughput = throughputSignals.filter((signal) => signal.sequence <= state.cursor);
  const maxTps = Math.max(...throughputSignals.map((signal) => signal.tokensPerSecond), 1);
  drawPolyline(
    visibleThroughput.map((signal) => [
      xForStep(signal.step, maxStep, left, width, rightPad),
      yInBand(signal.tokensPerSecond, maxTps, 132, 190),
    ]),
    "throughput-line",
  );

  const visiblePressure = pressureSamples.filter((sample) => sample.sequence <= state.cursor);
  drawPolyline(
    visiblePressure.map((sample) => [
      xForStep(sample.step, maxStep, left, width, rightPad),
      yInBand(sample.reservedBlocks, state.model.maxReservedBlocks, 132, 190),
    ]),
    "pressure-line",
  );

  legend(82, 208, "batch", "batch-swatch");
  legend(164, 208, "waiting", "waiting-swatch");
  legend(260, 208, "tok/s", "throughput-swatch");
  legend(346, 208, "reserved", "pressure-swatch");
}

function chartFrame(width, top, bottom, label) {
  line(els.signalSvg, 72, bottom, width - 42, bottom, "axis");
  text(els.signalSvg, 18, top + 8, label, "chart-label");
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

function xForStep(step, maxStep, left, width, rightPad = 42) {
  const span = Math.max(maxStep, 1);
  return left + (step / span) * (width - left - rightPad);
}

function yInBand(value, maxValue, top, bottom) {
  const ratio = maxValue === 0 ? 0 : value / maxValue;
  return bottom - ratio * (bottom - top);
}

function resetSvg(svg, width, height) {
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.replaceChildren();
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
  node.setAttribute("rx", 6);
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
  els.signalSvg.append(node);
}

function legend(x, y, label, className) {
  rect(els.signalSvg, x, y - 10, 12, 12, className);
  text(els.signalSvg, x + 18, y, label, "legend-label");
}

function setStatus(message, isError) {
  els.status.textContent = message;
  els.status.classList.toggle("is-error", isError);
}

function displaySourceName(sourceName) {
  return sourceName.split(/[\\/]/).at(-1) || sourceName;
}
