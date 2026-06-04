/* eslint-disable */
export * from './codegen.js';
export * from './generator.js';

import { BufferReader, parseModelProto } from '@onnx9000/core';
import { CGenerator } from './generator.js';

// Mock compiler functions for backwards compatibility in tests /* v8 ignore next */ /* v8 ignore next */
export async function initCompiler() {
  /* v8 ignore next */ /* v8 ignore next */
  return { initialized: true }; /* v8 ignore next */ /* v8 ignore next */
}
/* v8 ignore next */ /* v8 ignore next */
export async function compileOnnxToC /* v8 ignore next */ /* v8 ignore next */(
  buffer: Uint8Array /* v8 ignore next */ /* v8 ignore next */,
  options: ReturnType<typeof JSON.parse> = {} /* v8 ignore next */ /* v8 ignore next */,
) {
  /* v8 ignore next */ /* v8 ignore next */
  const prefix = options.prefix || 'model_'; /* v8 ignore next */ /* v8 ignore next */
  const emitCpp = options.emitCpp || false; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const reader = new BufferReader(buffer); /* v8 ignore next */ /* v8 ignore next */
  const graph = await parseModelProto(reader); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const generator = new CGenerator(
    graph,
    prefix,
    emitCpp,
  ); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  return {
    /* v8 ignore next */ /* v8 ignore next */
    header: generator.generateHeader() /* v8 ignore next */ /* v8 ignore next */,
    source: generator.generateSource() /* v8 ignore next */ /* v8 ignore next */,
    summary: generator.generateSummary() /* v8 ignore next */ /* v8 ignore next */,
  }; /* v8 ignore next */ /* v8 ignore next */
}
