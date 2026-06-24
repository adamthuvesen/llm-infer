export const TRACE_SCHEMA_VERSION = 2;

const KNOWN_EVENTS = new Set([
  "request_admitted",
  "prefill_chunk_started",
  "prefill_chunk_progress",
  "decode_step",
  "request_finished",
  "batch_size_changed",
  "tokens_per_second_sampled",
]);

const KNOWN_TOKEN_SOURCES = new Set(["prefill", "decode", "speculative"]);

export function parseJsonlTrace(text) {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length === 0) {
    throw new Error("Trace is empty.");
  }

  const events = lines.map((line, index) => {
    let event;
    try {
      event = JSON.parse(line);
    } catch (error) {
      throw new Error(`Line ${index + 1} is not valid JSON: ${error.message}`);
    }
    validateEvent(event, index + 1);
    return event;
  });

  const sequences = new Set();
  for (const event of events) {
    if (sequences.has(event.sequence)) {
      throw new Error(`Duplicate sequence ${event.sequence}.`);
    }
    sequences.add(event.sequence);
  }

  return [...events].sort((a, b) => a.sequence - b.sequence);
}

export function buildTraceModel(events) {
  const requests = new Map();
  const batchSignals = [];
  const throughputSignals = [];
  const pressureSamples = [];
  const active = new Set();
  const cachedTokens = new Map();
  const generatedTokens = new Map();

  const ensureRequest = (requestId) => {
    if (!requestId) {
      return null;
    }
    if (!requests.has(requestId)) {
      requests.set(requestId, {
        requestId,
        firstSequence: Number.POSITIVE_INFINITY,
        admittedStep: null,
        promptTokens: null,
        maxNewTokens: null,
        reservedBlocks: 0,
        prefixGroupId: null,
        chunks: [],
        decodes: [],
        finish: null,
      });
    }
    return requests.get(requestId);
  };

  const distributeDecode = (event) => {
    const requestIds = Array.isArray(event.request_ids) ? event.request_ids : [];
    const tokenIds = Array.isArray(event.token_ids) ? event.token_ids : [];
    if (requestIds.length === 1) {
      return [{ requestId: requestIds[0], tokenIds }];
    }
    return requestIds.map((requestId, index) => ({
      requestId,
      tokenIds: tokenIds[index] === undefined ? [] : [tokenIds[index]],
    }));
  };

  const samplePressure = (event) => {
    let reservedBlocks = 0;
    let logicalTokens = 0;
    for (const requestId of active) {
      const request = requests.get(requestId);
      reservedBlocks += request?.reservedBlocks ?? 0;
      logicalTokens += (cachedTokens.get(requestId) ?? 0) + (generatedTokens.get(requestId) ?? 0);
    }
    pressureSamples.push({
      sequence: event.sequence,
      step: event.step,
      reservedBlocks,
      logicalTokens,
      activeRequests: active.size,
    });
  };

  for (const event of events) {
    if (event.request_id) {
      const request = ensureRequest(event.request_id);
      request.firstSequence = Math.min(request.firstSequence, event.sequence);
    }

    if (event.event === "request_admitted") {
      const request = ensureRequest(event.request_id);
      Object.assign(request, {
        firstSequence: event.sequence,
        admittedStep: event.step,
        promptTokens: event.prompt_tokens,
        maxNewTokens: event.max_new_tokens,
        reservedBlocks: event.reserved_blocks ?? 0,
        prefixGroupId: event.prefix_group_id ?? null,
      });
      active.add(event.request_id);
      cachedTokens.set(event.request_id, 0);
      generatedTokens.set(event.request_id, 0);
    }

    if (event.event === "prefill_chunk_progress") {
      const request = ensureRequest(event.request_id);
      request.chunks.push({
        sequence: event.sequence,
        step: event.step,
        startPos: event.start_pos,
        endPos: event.end_pos,
        cachedTokens: event.cached_tokens,
        totalPromptTokens: event.total_prompt_tokens,
        completed: event.completed === true,
      });
      cachedTokens.set(event.request_id, event.cached_tokens ?? cachedTokens.get(event.request_id) ?? 0);
    }

    if (event.event === "decode_step") {
      for (const item of distributeDecode(event)) {
        const request = ensureRequest(item.requestId);
        request.firstSequence = Math.min(request.firstSequence, event.sequence);
        request.decodes.push({
          sequence: event.sequence,
          step: event.step,
          tokenIds: item.tokenIds,
          tokenSource: event.token_source ?? "decode",
          batchSize: event.batch_size ?? null,
          tokensEmitted: event.tokens_emitted ?? item.tokenIds.length,
        });
        generatedTokens.set(item.requestId, (generatedTokens.get(item.requestId) ?? 0) + item.tokenIds.length);
      }
    }

    if (event.event === "request_finished") {
      const request = ensureRequest(event.request_id);
      request.finish = {
        sequence: event.sequence,
        step: event.step,
        tokenIds: Array.isArray(event.token_ids) ? event.token_ids : [],
        generatedTokens: event.generated_tokens ?? null,
        reason: event.reason ?? "unknown",
      };
      generatedTokens.set(event.request_id, event.generated_tokens ?? generatedTokens.get(event.request_id) ?? 0);
      active.delete(event.request_id);
    }

    if (event.event === "batch_size_changed") {
      batchSignals.push({
        sequence: event.sequence,
        step: event.step,
        batchSize: event.batch_size ?? 0,
        previousBatchSize: event.previous_batch_size ?? 0,
        waiting: event.waiting ?? 0,
      });
    }

    if (event.event === "tokens_per_second_sampled") {
      throughputSignals.push({
        sequence: event.sequence,
        step: event.step,
        tokensPerSecond: event.tokens_per_second ?? 0,
        tokensEmitted: event.tokens_emitted ?? 0,
        totalGeneratedTokens: event.total_generated_tokens ?? 0,
        elapsedSeconds: event.elapsed_seconds ?? 0,
      });
    }

    samplePressure(event);
  }

  const requestList = [...requests.values()].sort((a, b) => a.firstSequence - b.firstSequence);
  const maxStep = Math.max(...events.map((event) => event.step), 0);
  const maxSequence = Math.max(...events.map((event) => event.sequence), 0);
  const maxReservedBlocks = Math.max(...pressureSamples.map((sample) => sample.reservedBlocks), 1);
  const maxLogicalTokens = Math.max(...pressureSamples.map((sample) => sample.logicalTokens), 1);

  return {
    events,
    requestList,
    batchSignals,
    throughputSignals,
    pressureSamples,
    maxStep,
    maxSequence,
    maxReservedBlocks,
    maxLogicalTokens,
  };
}

export function eventLabel(event) {
  const stem = event.event.replaceAll("_", " ");
  if (event.request_id) {
    return `${stem} · ${event.request_id}`;
  }
  if (Array.isArray(event.request_ids) && event.request_ids.length > 0) {
    return `${stem} · ${event.request_ids.join(", ")}`;
  }
  return stem;
}

function validateEvent(event, lineNumber) {
  if (event === null || typeof event !== "object" || Array.isArray(event)) {
    throw new Error(`Line ${lineNumber} must be a JSON object.`);
  }
  if (event.schema_version !== TRACE_SCHEMA_VERSION) {
    throw new Error(
      `Line ${lineNumber} has schema_version ${event.schema_version}; expected ${TRACE_SCHEMA_VERSION}.`,
    );
  }
  if (!KNOWN_EVENTS.has(event.event)) {
    throw new Error(`Line ${lineNumber} has unknown event ${JSON.stringify(event.event)}.`);
  }
  if (!Number.isInteger(event.sequence) || event.sequence < 1) {
    throw new Error(`Line ${lineNumber} must have a positive integer sequence.`);
  }
  if (!Number.isInteger(event.step) || event.step < 0) {
    throw new Error(`Line ${lineNumber} must have a non-negative integer step.`);
  }
  if (event.request_id !== undefined && typeof event.request_id !== "string") {
    throw new Error(`Line ${lineNumber} request_id must be a string when present.`);
  }
  if (
    event.request_ids !== undefined &&
    (!Array.isArray(event.request_ids) ||
      event.request_ids.some((requestId) => typeof requestId !== "string"))
  ) {
    throw new Error(`Line ${lineNumber} request_ids must be an array of strings when present.`);
  }
  if (event.token_source !== undefined && !KNOWN_TOKEN_SOURCES.has(event.token_source)) {
    throw new Error(`Line ${lineNumber} has unknown token_source ${JSON.stringify(event.token_source)}.`);
  }
}
