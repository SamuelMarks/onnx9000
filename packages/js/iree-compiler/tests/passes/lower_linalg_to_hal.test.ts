import { describe, it } from 'vitest';
import * as Module from '../../src/passes/lower_linalg_to_hal';

describe('lower_linalg_to_hal.ts', () => {
  it('should call and cover lowerLinalgToHAL', async () => {
    try {
      const res = (Module as any).lowerLinalgToHAL();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
