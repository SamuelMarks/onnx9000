import { describe, it, expect } from 'vitest';
import { pipeline } from '../src/pipelines/index.js';

describe('pipelines', () => {
  it('should run pipeline', async () => {
    const pipe = await pipeline('text-classification');
    const res = await pipe('hello');
    expect(res).toBeDefined();
    expect(res[0].label).toBe('positive');
  });
});
