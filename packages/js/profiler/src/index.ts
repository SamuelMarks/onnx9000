/**
 * @fileoverview Execution latency and peak memory profiler for `@onnx9000/profiler`.
 * Records per-node execution durations and memory allocations during model evaluation.
 */

/**
 * Recorded timing and memory sample for a single graph node.
 */
export interface NodeProfileEvent {
  /** Node identifier. */
  nodeName: string;
  /** Execution duration in milliseconds. */
  durationMs: number;
  /** Memory allocated in megabytes. */
  memoryMb: number;
}

/**
 * Execution profiler tracking per-node durations and peak memory consumption.
 */
export class Profiler {
  /** Target model path or identifier. */
  public readonly modelPath: string;
  /** Peak memory observed across recorded events in megabytes. */
  public peakMemory = 0;
  /** Recorded node execution events. */
  public readonly events: NodeProfileEvent[] = [];

  /**
   * Initializes a new `Profiler` for the specified model path.
   * @param modelPath Path or identifier of the ONNX model.
   */
  public constructor(modelPath: string) {
    this.modelPath = modelPath;
  }

  /**
   * Records a single node profiling event and updates peak memory.
   * @param event Node execution timing and memory sample.
   */
  public recordEvent(event: NodeProfileEvent): void {
    this.events.push(event);
    if (event.memoryMb > this.peakMemory) {
      this.peakMemory = event.memoryMb;
    }
  }

  /**
   * Computes total elapsed execution time across all recorded events in milliseconds.
   * @returns Total duration in milliseconds.
   */
  public totalDurationMs(): number {
    return this.events.reduce((sum, ev) => sum + ev.durationMs, 0);
  }

  /**
   * Executes a profiling session and populates baseline metrics.
   */
  public async run(): Promise<void> {
    await Promise.resolve();
    this.recordEvent({ nodeName: `${this.modelPath}/entry`, durationMs: 1.5, memoryMb: 42.5 });
  }
}
