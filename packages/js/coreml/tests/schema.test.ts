import { describe, it, expect } from 'vitest';
import { emitModel } from '../src/emitter.js';
import { parseModel } from '../src/schema.js';
import { BufferReader } from '@onnx9000/core';

describe('Schema Emitter & Parser Coverage', () => {
  it('covers full roundtrip of model properties', async () => {
    const model = {
      specificationVersion: 2,
      description: {
        input: [{ name: 'in1', shortDescription: 'desc1' }],
        output: [{ name: 'out1', shortDescription: 'desc2' }],
        metadata: {
          shortDescription: 'meta1',
          versionString: '1.0',
          author: 'author',
          license: 'mit',
        },
      },
      neuralNetwork: { layers: [] },
    };

    const bytes = emitModel(model as Object);
    expect(bytes).toBeInstanceOf(Uint8Array);

    const model2 = {
      specificationVersion: 3,
      description: { input: [], output: [] },
      mlProgram: { version: 1, functions: {} },
    };
    const bytes2 = emitModel(model2 as Object);
    expect(bytes2).toBeInstanceOf(Uint8Array);

    const reader = new BufferReader(bytes);
    const parsed = await parseModel(reader);
    expect(parsed.specificationVersion).toBe(2);
  });

  it('covers unmapped fields in parser', async () => {
    const buf = new Uint8Array([(1 << 3) | 0, 4, (99 << 3) | 0, 1]);
    const parsed = await parseModel(new BufferReader(buf));
    expect(parsed.specificationVersion).toBe(4);
  });
});
