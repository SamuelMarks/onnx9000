import * as fs from 'fs';
import { JsonValue } from './tfjs-parser.js';

/**
 * Represents a Chrome Trace event for profiling.
 */
export interface TraceEvent {
  /** Event name. */
  name: string;
  /** Event category. */
  cat: string;
  /** Phase (B: Begin, E: End, C: Counter). */
  ph: string;
  /** Timestamp in microseconds. */
  ts: number;
  /** Process ID. */
  pid: number;
  /** Thread ID. */
  tid: number;
  /** Optional arguments. */
  args?: Record<string, JsonValue>;
}

/**
 * Exporter for Chrome Trace format (JSON).
 */
export class ChromeTraceExporter {
  private events: TraceEvent[] = [];

  /**
   * Start a timed event.
   * @param name Event name.
   * @param category Event category.
   */
  public startEvent(name: string, category: string = 'keras_to_onnx') {
    this.events.push({
      name,
      cat: category,
      ph: 'B',
      ts: Date.now() * 1000,
      pid: 1,
      tid: 1,
    });
  }

  /**
   * End a timed event.
   * @param name Event name.
   * @param category Event category.
   */
  public endEvent(name: string, category: string = 'keras_to_onnx') {
    this.events.push({
      name,
      cat: category,
      ph: 'E',
      ts: Date.now() * 1000,
      pid: 1,
      tid: 1,
    });
  }

  /**
   * Record a memory allocation counter.
   * @param bytes Number of bytes allocated.
   */
  public recordMemory(bytes: number) {
    this.events.push({
      name: 'MemoryAlloc',
      cat: 'memory',
      ph: 'C',
      ts: Date.now() * 1000,
      pid: 1,
      tid: 1,
      args: { allocated: bytes },
    });
  }

  /**
   * Save the trace to a file.
   * @param path File path.
   */
  public save(path: string) {
    fs.writeFileSync(path, JSON.stringify(this.events, null, 2));
  }
}

/**
 * Validate that two sets of outputs are within mathematical tolerance.
 * @param kerasOutputs Expected outputs.
 * @param onnxOutputs Actual outputs.
 * @param tolerance Maximum absolute error allowed.
 * @returns True if within tolerance.
 */
export function validateMathematicalTolerance(
  kerasOutputs: Float32Array,
  onnxOutputs: Float32Array,
  tolerance: number = 1e-4,
): boolean {
  if (kerasOutputs.length !== onnxOutputs.length) return false;
  for (let i = 0; i < kerasOutputs.length; i++) {
    const kerasVal = kerasOutputs[i];
    const onnxVal = onnxOutputs[i];
    if (kerasVal !== undefined && onnxVal !== undefined) {
      if (Math.abs(kerasVal - onnxVal) > tolerance) {
        return false;
      }
    }
  }
  return true;
}
