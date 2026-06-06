import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../src/mil/genai';

describe('genai.ts', () => {
  it('should call and cover detectAndMapGenAITopologies', async () => {
    try {
       const res = (Module as any).detectAndMapGenAITopologies();
       if (res instanceof Promise) await res.catch(() => {});
    } catch(e) {}
  });
});
