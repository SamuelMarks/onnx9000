/**
 * @fileoverview Core entry point for the `@onnx9000/core` package.
 * Exports IR structures, binary parsers, shape inference, operators, and custom exceptions.
 */
export * from './exceptions.js';
export * from './ir/graph.js';
export * from './ir/node.js';
export * from './ir/tensor.js';
export * from './parser/magic.js';
export * from './parser/onnx.js';
export * from './parser/onnx_writer.js';
export * from './parser/protobuf.js';
export * from './parser/safetensors.js';
export * from './parser/safetensors.node.js';
export * from './shape_inference/infer.js';

import type { Graph } from './ir/graph.js';
import { parseModelProto } from './parser/onnx.js';
import { serializeModelProto } from './parser/onnx_writer.js';
import { BufferReader } from './parser/protobuf.js';

/**
 * Loads an ONNX model from an ArrayBuffer or Uint8Array.
 * @param buffer The serialized ONNX model binary data.
 * @returns A Promise resolving to the parsed `Graph`.
 */
export async function load(buffer: ArrayBuffer | Uint8Array): Promise<Graph> {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  const reader = new BufferReader(bytes);
  return await parseModelProto(reader);
}

/**
 * Serializes a `Graph` into an ONNX binary `ArrayBuffer`.
 * @param graph The `Graph` instance to serialize.
 * @returns A Promise resolving to the serialized `ArrayBuffer`.
 */
export async function save(graph: Graph): Promise<ArrayBuffer> {
  const bytes = serializeModelProto(graph);
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return buffer as ArrayBuffer;
}

export * from './checker.js';
export * from './codegen/pytorch.js';
export * from './macros.js';
export * from './models/index.js';
export type { OpImplementation } from './ops/registry.js';
export { globalRegistry, register_op } from './ops/registry.js';
export * from './primitives.js';
export * from './sharding.js';
export * from './sparse.js';
