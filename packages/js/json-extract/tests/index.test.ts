import { describe, it, expect } from 'vitest';
import { extractJson, createOnnxJsonReplacer } from '../src/index.js';
import { Graph, Tensor } from '@onnx9000/core';

describe('JSON Extract SDK', () => {
  it('should stringify a basic graph without buffers', () => {
    const graph = new Graph('TestGraph');
    const json = extractJson(graph);
    expect(typeof json).toBe('string');
    const parsed = JSON.parse(json);
    expect(parsed.name).toBe('TestGraph');
  });

  it('should handle BigInt conversion correctly', () => {
    const obj = { val: 123n };
    const json = JSON.stringify(obj, createOnnxJsonReplacer());
    expect(json).toBe('{"val":"123n"}');
  });

  it('should handle ArrayBuffer and ArrayBufferView when dropBuffers is true', () => {
    const buffer = new ArrayBuffer(16);
    const view = new Uint8Array(8);
    const obj = { buf: buffer, v: view };

    const json = JSON.stringify(obj, createOnnxJsonReplacer({ dropBuffers: true }));
    const parsed = JSON.parse(json);
    expect(parsed.buf).toBe('[Buffer: 16 bytes]');
    expect(parsed.v).toBe('[Buffer: 8 bytes]');
  });

  it('should allow custom bufferReplacer', () => {
    const buffer = new ArrayBuffer(16);
    const view = new Uint8Array(8);
    const obj = { buf: buffer, v: view };

    const replacer = (val: ArrayBuffer | ArrayBufferView) => `[Dropped: ${val.byteLength}]`;
    const json = JSON.stringify(
      obj,
      createOnnxJsonReplacer({ dropBuffers: true, bufferReplacer: replacer }),
    );
    const parsed = JSON.parse(json);
    expect(parsed.buf).toBe('[Dropped: 16]');
    expect(parsed.v).toBe('[Dropped: 8]');
  });

  it('should keep buffers when dropBuffers is false', () => {
    const view = new Uint8Array([1, 2, 3]);
    const obj = { v: view };

    // JSON.stringify natively converts Uint8Array to an object-like map or array, depending on platform.
    const json = JSON.stringify(obj, createOnnxJsonReplacer({ dropBuffers: false }));
    const parsed = JSON.parse(json);
    // Usually { "0": 1, "1": 2, "2": 3 } natively
    expect(parsed.v).toBeDefined();
    expect(typeof parsed.v).toBe('object');
  });

  it('should integrate correctly with a Graph containing a Tensor (which has buffer data)', () => {
    const graph = new Graph('GraphWithTensor');
    const t = new Tensor('w', [2, 2], 'float32', true);
    // Force a dummy buffer
    t.data = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    graph.initializers.push(t);

    const json = extractJson(graph);
    const parsed = JSON.parse(json);

    expect(parsed.name).toBe('GraphWithTensor');
    expect(parsed.initializers[0].name).toBe('w');
    // Because Tensor.data is a Float32Array
    expect(parsed.initializers[0].data).toBe('[Buffer: 16 bytes]');
  });
});
