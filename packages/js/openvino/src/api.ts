/**
 * @fileoverview api.ts
 * Provides api functionality for the openvino package.
 */
import type { Graph } from '@onnx9000/core';
import { OpenVinoExporter, type OpenVinoExportOptions } from './exporter.js';

export function exportModel(
  onnxModel: Graph,
  options?: OpenVinoExportOptions,
): { xml: string; bin: Uint8Array } {
  const exporter = new OpenVinoExporter(onnxModel, options);
  return exporter.export();
}
