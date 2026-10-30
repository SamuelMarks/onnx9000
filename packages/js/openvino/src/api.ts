/**
 * @fileoverview api.ts
 * Provides api functionality for the openvino package.
 */
import type { Graph } from '@onnx9000/core';
import { OpenVinoExporter, type OpenVinoExportOptions } from './exporter.js';

/**
 * Executes `exportModel`.
 * @param onnxModel Parameter `onnxModel`.
 * @param options Parameter `options`.
 * @returns Result of `exportModel`.
 */
export function exportModel(
  onnxModel: Graph,
  options?: OpenVinoExportOptions,
): { xml: string; bin: Uint8Array } {
  const exporter = new OpenVinoExporter(onnxModel, options);
  return exporter.export();
}
