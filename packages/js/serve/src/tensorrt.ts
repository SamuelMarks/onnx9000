import { Graph } from '@onnx9000/core';

export async function createTensorRTSession(graph: Graph): Promise<any> {
  // Use dynamic import so it doesn't break environments without FFI (e.g. edge workers)
  try {
    const trt = await import('@onnx9000/tensorrt');
    return new trt.TensorRTProvider(graph);
  } catch (e) {
    throw new Error('TensorRT provider requires Node.js and ffi-napi: ' + e);
  }
}
