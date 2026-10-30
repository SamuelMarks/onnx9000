/**
 * @fileoverview Web Worker script for compiling ONNX models to C code off the main UI thread.
 * Prevents freezing the UI during compilation.
 */
import { compileOnnxToC } from '@onnx9000/c-compiler';

/** Interface `WorkerCompileResponse` definition. */
export interface WorkerCompileResponse {
  /** Property `header`. */
  header?: string;
  /** Property `source`. */
  source?: string;
  /** Property `summary`. */
  summary?: string;
  /** Property `arenaSize`. */
  arenaSize?: number;
  /** Property `error`. */
  error?: string;
}

export const handleWorkerMessage = async (
  e: MessageEvent,
  postMessage: (msg: WorkerCompileResponse) => void,
) => {
  const { buffer, options } = e.data;

  try {
    const result = await compileOnnxToC(buffer, options);
    const arenaSize = 250000;

    postMessage({
      header: result.header,
      source: result.source,
      summary: result.summary,
      arenaSize: arenaSize,
    });
  } catch (_err) {
    const err = _err instanceof Error ? _err : new Error(String(_err));
    postMessage({ error: err.message });
  }
};

if (typeof self !== 'undefined' && self.postMessage) {
  self.onmessage = (e) => handleWorkerMessage(e, self.postMessage.bind(self));
}
