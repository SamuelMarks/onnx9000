/**
 * @fileoverview batcher.ts
 * Provides batcher functionality for the serve package.
 */
export interface BatchRequest {
  /** Property `id`. */
  id: string;
  /** Property `payload`. */
  payload: ReturnType<typeof JSON.parse>;
  /** Property `resolve`. */
  resolve: (response: ReturnType<typeof JSON.parse>) => void;
  /** Property `reject`. */
  reject: (error: Error) => void;
  /** Property `queuedAt`. */
  queuedAt: number;
  /** Property `priority`. */
  priority: number; // Higher is better
}

/** Represents the `DynamicBatcher` class. */
export class DynamicBatcher {
  /** Property `maxBatchSize` of `DynamicBatcher`. */
  public maxBatchSize: number = 8;
  /** Property `batchTimeoutMs` of `DynamicBatcher`. */
  public batchTimeoutMs: number = 10;

  /** Property `queue` of `DynamicBatcher`. */
  private queue: BatchRequest[] = [];
  /** Property `timeoutHandle` of `DynamicBatcher`. */
  private timeoutHandle: ReturnType<typeof JSON.parse> = null;

  /**
   * Initializes a new `DynamicBatcher` instance.
   * @param executeBatch Parameter `executeBatch`.
   * @param options Parameter `options`.
   */
  constructor(
    private executeBatch: (
      batch: BatchRequest[],
      concatenatedInputs: ReturnType<typeof JSON.parse>,
    ) => Promise<ReturnType<typeof JSON.parse>[]>,
    options?: { maxBatchSize?: number; batchTimeoutMs?: number },
  ) {
    if (options?.maxBatchSize) this.maxBatchSize = options.maxBatchSize;
    if (options?.batchTimeoutMs) this.batchTimeoutMs = options.batchTimeoutMs;
  }

  /**
   * Executes `add` on `DynamicBatcher`.
   * @param payload Parameter `payload`.
   * @param priority Parameter `priority`.
   * @returns Result of `add`.
   */
  public async add(
    payload: ReturnType<typeof JSON.parse>,
    priority: number = 0,
  ): Promise<ReturnType<typeof JSON.parse>> {
    return new Promise((resolve, reject) => {
      const request: BatchRequest = {
        id: crypto.randomUUID(),
        payload,
        resolve,
        reject,
        queuedAt: Date.now(),
        priority,
      };

      this.queue.push(request);

      if (this.queue.length >= this.maxBatchSize) {
        this.flush();
      } else if (!this.timeoutHandle) {
        this.timeoutHandle = setTimeout(() => {
          this.flush();
        }, this.batchTimeoutMs);
      }
    });
  }

  /**
   * Executes `flush` on `DynamicBatcher`.
   * @returns Result of `flush`.
   */
  private flush() {
    if (this.timeoutHandle) {
      clearTimeout(this.timeoutHandle);
      this.timeoutHandle = null;
    }

    if (this.queue.length === 0) return;

    // 42. Implement Priority Queueing (prioritizing premium user requests over standard).
    // Sort by priority descending, then queuedAt ascending
    this.queue.sort((a, b) => {
      if (a.priority !== b.priority) {
        return b.priority - a.priority;
      }
      return a.queuedAt - b.queuedAt;
    });

    // 41. Ensure strict ordering of responses matching the incoming queue exactly.
    const activeBatch = this.queue.splice(0, this.maxBatchSize);

    const startTime = (activeBatch[0] as BatchRequest).queuedAt;
    console.debug(`Batched ${activeBatch.length} requests in ${Date.now() - startTime}ms`);

    setTimeout(async () => {
      try {
        // 37. Implement tensor concatenation across the batch dimension (`Axis 0`) dynamically.
        // 38. Pad variable-length sequence inputs automatically (e.g., text inputs) within the batch.
        // 39. Generate dynamic `attention_mask` tensors for padded sequences securely.
        const concatenatedInputs = this.prepareBatchInputs(activeBatch);

        // Execute
        const outputs = await this.executeBatch(activeBatch, concatenatedInputs);

        // 40. Split the single ONNX execution output back into isolated HTTP response promises.
        // 41. Strict ordering
        for (let i = 0; i < activeBatch.length; i++) {
          activeBatch[i]?.resolve(outputs[i]);
        }
      } catch (_err) {
        const err = _err instanceof Error ? _err : new Error(String(_err));
        // 43. Handle batching failures by isolating the failure and re-executing the valid subset.
        // (Simplified handling for now: reject all)
        for (const req of activeBatch) {
          req.reject(err);
        }
      }
    }, 0);
  }

  /**
   * Executes `prepareBatchInputs` on `DynamicBatcher`.
   * @param batch Parameter `batch`.
   * @returns Result of `prepareBatchInputs`.
   */
  private prepareBatchInputs(batch: BatchRequest[]): ReturnType<typeof JSON.parse> {
    const hasSequences = batch.some((b) => Array.isArray(b.payload?.input_ids));
    if (!hasSequences) {
      return {
        batch_size: batch.length,
        items: batch.map((b) => b.payload),
      };
    }

    let maxLen = 0;
    for (const req of batch) {
      const len = req.payload?.input_ids?.length || 0;
      if (len > maxLen) maxLen = len;
    }

    const paddedInputIds: number[][] = [];
    const attentionMasks: number[][] = [];

    for (const req of batch) {
      const seq: number[] = req.payload?.input_ids || [];
      const padLen = maxLen - seq.length;

      const paddedSeq = [...seq, ...new Array(padLen).fill(0)];
      const mask = [...new Array(seq.length).fill(1), ...new Array(padLen).fill(0)];

      paddedInputIds.push(paddedSeq);
      attentionMasks.push(mask);
    }

    return {
      input_ids: paddedInputIds,
      attention_mask: attentionMasks,
    };
  }
}
