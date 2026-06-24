export const TRACE_SCHEMA_VERSION = 3;

const KNOWN_EVENTS = new Set([
  "request_admitted",
  "prefill_chunk_started",
  "prefill_chunk_progress",
  "decode_step",
  "block_allocated",
  "block_freed",
  "request_preempted",
  "request_resumed",
  "request_finished",
  "batch_size_changed",
  "tokens_per_second_sampled",
]);

const KNOWN_PREEMPT_REASONS = new Set(["kv_pressure"]);

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
  const heldBlocks = new Map();
  // Pool occupancy reported by the allocator itself (block_allocated/block_freed carry the
  // post-change totals), so the KV wall can show blocks actually held, not just reserved.
  let poolUsed = 0;
  let poolFree = null;

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
        blockEvents: [],
        preempts: [],
        resumes: [],
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
      // Blocks physically held right now, summed from the per-request running counts the
      // allocator events drive. Shared prefix blocks are counted once, by their last owner.
      allocatedBlocks: poolUsed,
      poolFree,
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

    if (event.event === "block_allocated" || event.event === "block_freed") {
      const request = ensureRequest(event.request_id);
      const blockCount = event.block_count ?? (Array.isArray(event.block_ids) ? event.block_ids.length : 0);
      const signed = event.event === "block_allocated" ? blockCount : -blockCount;
      if (request) {
        request.blockEvents.push({
          sequence: event.sequence,
          step: event.step,
          kind: event.event,
          blockCount,
          blockIds: Array.isArray(event.block_ids) ? event.block_ids : [],
        });
        heldBlocks.set(event.request_id, Math.max(0, (heldBlocks.get(event.request_id) ?? 0) + signed));
      }
      // Prefer the allocator's own post-change totals; fall back to the running sum.
      poolUsed = event.pool_used ?? Math.max(0, poolUsed + signed);
      poolFree = event.pool_free ?? poolFree;
    }

    if (event.event === "request_preempted") {
      const request = ensureRequest(event.request_id);
      request.preempts.push({
        sequence: event.sequence,
        step: event.step,
        reason: event.preempt_reason ?? "kv_pressure",
        freedBlocks: event.block_count ?? 0,
        generatedTokens: event.generated_tokens ?? generatedTokens.get(event.request_id) ?? 0,
        poolUsed: event.pool_used ?? null,
        poolFree: event.pool_free ?? null,
      });
      // Evicted: its KV is dropped (the matching block_freed already returned the blocks to the
      // pool), so it no longer occupies the wall, but it keeps its generated tokens for recompute.
      cachedTokens.set(event.request_id, 0);
      active.delete(event.request_id);
    }

    if (event.event === "request_resumed") {
      const request = ensureRequest(event.request_id);
      request.resumes.push({
        sequence: event.sequence,
        step: event.step,
        cachedTokens: event.cached_tokens ?? 0,
        generatedTokens: event.generated_tokens ?? generatedTokens.get(event.request_id) ?? 0,
      });
      // Re-admitted and rebuilding by recompute — back on the wall as its prefill replays.
      cachedTokens.set(event.request_id, event.cached_tokens ?? 0);
      active.add(event.request_id);
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
  const maxAllocatedBlocks = Math.max(...pressureSamples.map((sample) => sample.allocatedBlocks), 0);
  // Pool capacity = peak blocks ever in use simultaneously. With pool_free present we know the
  // true size; otherwise fall back to the reservation wall so the grid stays meaningful.
  const poolFromFree = Math.max(
    ...pressureSamples.map((sample) =>
      sample.poolFree === null || sample.poolFree === undefined ? 0 : sample.allocatedBlocks + sample.poolFree,
    ),
    0,
  );
  const poolCapacity = poolFromFree > 0 ? poolFromFree : maxReservedBlocks;
  const hasBlockLifecycle = maxAllocatedBlocks > 0;

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
    maxAllocatedBlocks,
    poolCapacity,
    hasBlockLifecycle,
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
  if (event.preempt_reason !== undefined && !KNOWN_PREEMPT_REASONS.has(event.preempt_reason)) {
    throw new Error(`Line ${lineNumber} has unknown preempt_reason ${JSON.stringify(event.preempt_reason)}.`);
  }
  if (event.event === "block_allocated" || event.event === "block_freed") {
    validateBlockEvent(event, lineNumber);
  }
}

function validateBlockEvent(event, lineNumber) {
  if (!Array.isArray(event.block_ids) || event.block_ids.some((id) => !Number.isInteger(id) || id < 0)) {
    throw new Error(`Line ${lineNumber} ${event.event} must have block_ids as non-negative integers.`);
  }
  if (!Number.isInteger(event.block_count) || event.block_count !== event.block_ids.length) {
    throw new Error(`Line ${lineNumber} ${event.event} block_count must match block_ids length.`);
  }
  for (const field of ["pool_used", "pool_free"]) {
    if (!Number.isInteger(event[field]) || event[field] < 0) {
      throw new Error(`Line ${lineNumber} ${event.event} must have a non-negative integer ${field}.`);
    }
  }
}
