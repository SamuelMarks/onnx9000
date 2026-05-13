import { Graph } from '@onnx9000/core';

export interface JsonExtractOptions {
  /** Drop large buffer payloads like ArrayBuffer/Uint8Array */
  dropBuffers?: boolean;
  /** Replacer for buffers if dropBuffers is true (default: "[Buffer: X bytes]") */
  bufferReplacer?: (value: ArrayBufferView | ArrayBuffer) => string;
  /** Formatting spaces for JSON.stringify (default: 2) */
  spaces?: number;
}

/**
 * Custom replacer to handle buffers and BigInts safely when stringifying ONNX AST graphs.
 */
export function createOnnxJsonReplacer(options: JsonExtractOptions = {}) {
  const { dropBuffers = true, bufferReplacer } = options;

  return function replacer(key: string, value: unknown): unknown {
    if (dropBuffers) {
      if (value instanceof ArrayBuffer) {
        if (bufferReplacer) return bufferReplacer(value);
        return `[Buffer: ${String(value.byteLength)} bytes]`;
      }
      if (ArrayBuffer.isView(value)) {
        if (bufferReplacer) return bufferReplacer(value);
        return `[Buffer: ${String(value.byteLength)} bytes]`;
      }
    }

    if (typeof value === 'bigint') {
      return value.toString() + 'n';
    }

    return value;
  };
}

/**
 * Extracts a JSON string representation of an ONNX Graph AST.
 * Handles safe stringification of BigInts and optionally strips large buffer data.
 *
 * @param graph The @onnx9000/core Graph instance
 * @param options Extraction options
 * @returns Serialized JSON string
 */
export function extractJson(graph: Graph, options: JsonExtractOptions = {}): string {
  const { spaces = 2 } = options;
  return JSON.stringify(graph, createOnnxJsonReplacer(options), spaces);
}
