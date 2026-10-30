/**
 * @fileoverview tensorrt.ts
 * Provides tensorrt functionality for the serve package.
 */

import type { Graph } from '@onnx9000/core';
import { BackendError } from '@onnx9000/core';

/**
 * Executes `createTensorRTSession`.
 * @param graph Parameter `graph`.
 * @returns Result of `createTensorRTSession`.
 */
export async function createTensorRTSession(graph: Graph): Promise<ReturnType<typeof JSON.parse>> {
  // Use dynamic import so it doesn't break environments without FFI (e.g. edge workers)
  try {
    const trt = await import('@onnx9000/tensorrt');
    return new trt.TensorRTProvider(graph);
  } catch (e) {
    throw new BackendError(`TensorRT provider requires Node.js and ffi-napi: ${e}`);
  }
}
