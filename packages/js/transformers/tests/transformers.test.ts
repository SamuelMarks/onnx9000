import { describe, it, expect } from 'vitest';
import * as t from '../src/index.js';

describe('Transformers Exports', () => {
  it('should export pipelines', async () => {
    const pipe = await t.pipeline('text-generation');
    expect(pipe.task).toBe('text-generation');
    const res = await pipe('hello');
    expect(res).toEqual([{ generated_text: 'hello [GENERATED]' }]);

    await expect(t.pipeline('unknown')).rejects.toThrow('Unsupported task: unknown');
  });

  it('should export models', async () => {
    const model = await t.AutoModel.fromPretrained('dummy');
    expect(model).toBeInstanceOf(t.AutoModel);
  });

  it('should export tokenizers', async () => {
    const tok = await t.AutoTokenizer.fromPretrained('dummy');
    expect(tok.encode('hi')).toEqual([104]);
    expect(tok.decode([104])).toBe('h');
  });

  it('should export processors', async () => {
    const proc = await t.AutoProcessor.fromPretrained('dummy');
    expect(proc.process('image')).toEqual({ pixel_values: [0.5, 0.5] });
  });

  it('should export array api', () => {
    expect(t.ArrayAPI.add([1, 2], [3, 4])).toEqual([4, 6]);
  });
});

describe('ArrayAPI bounds coverage', () => {
  it('should handle b shorter than a', () => {
    const a = [1, 2, 3];
    const b = [1];
    const res = t.ArrayAPI.add(a, b);
    expect(res).toEqual([2, 2, 3]);
  });
});
