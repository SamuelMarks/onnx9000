/**
 * @fileoverview index.ts
 * Provides index functionality for the c-compiler package.
 */
export * from './codegen.js';
export * from './generator.js';

import { BufferReader, parseModelProto } from '@onnx9000/core';
import { CGenerator } from './generator.js';

// Mock compiler functions for backwards compatibility in tests
export async function initCompiler() {
  return { initialized: true };
}

export async function compileOnnxToC(
  buffer: Uint8Array,
  options: ReturnType<typeof JSON.parse> = {},
) {
  const prefix = options.prefix || 'model_';
  const emitCpp = options.emitCpp || false;

  const reader = new BufferReader(buffer);
  const graph = await parseModelProto(reader);

  const generator = new CGenerator(graph, prefix, emitCpp);

  return {
    header: generator.generateHeader(),
    source: generator.generateSource(),
    summary: generator.generateSummary(),
  };
}
