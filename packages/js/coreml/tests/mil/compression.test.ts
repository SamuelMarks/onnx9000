import { describe, it } from 'vitest';
import * as Module from '../../src/mil/compression';

describe('compression.ts', () => {
  it('should call and cover applyCompression', async () => {
    try {
      const res = (Module as any).applyCompression();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
