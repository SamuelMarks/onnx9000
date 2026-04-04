export * from './ir/graph.js';
export * from './ir/node.js';
export * from './ir/tensor.js';
export * from './parser/protobuf.js';
export * from './parser/onnx.js';
export * from './shape_inference/infer.js';
export * from './parser/magic.js';
export * from './parser/safetensors.js';
export * from './parser/safetensors.node.js';
export * from './parser/onnx_writer.js';

import { Graph } from './ir/graph.js';
import { BufferReader } from './parser/protobuf.js';
import { parseModelProto } from './parser/onnx.js';
import { serializeModelProto } from './parser/onnx_writer.js';

/**
 * Loads an ONNX model from an ArrayBuffer or Uint8Array.
 * @param buffer The model data
 * @returns A Promise that resolves to the parsed Graph
 */
export async function load(buffer: ArrayBuffer | Uint8Array): Promise<Graph> {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  const reader = new BufferReader(bytes);
  return await parseModelProto(reader);
}

/**
 * Saves a Graph to an ArrayBuffer in ONNX format.
 * @param graph The graph to save
 * @returns Serialized model data
 */
export async function save(graph: Graph): Promise<ArrayBuffer> {
  const bytes = serializeModelProto(graph);
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return buffer as ArrayBuffer;
}

export * from './sparse.js';
export * from './checker.js';
export * from './primitives.js';
export * from './models/index.js';

export * from './macros.js';

export * from './sharding.js';
