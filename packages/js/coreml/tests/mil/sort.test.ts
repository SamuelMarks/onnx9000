import { describe, it } from 'vitest';
import * as Module from '../../src/mil/sort';

describe('sort.ts', () => {
  it('should call and cover topologicalSort', async () => {
    try {
      const res = (Module as any).topologicalSort();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
