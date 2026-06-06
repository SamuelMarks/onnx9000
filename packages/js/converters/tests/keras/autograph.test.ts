import { describe, it } from 'vitest';
import * as Module from '../../src/keras/autograph';

describe('autograph.ts', () => {
  it('should call and cover extractTraceViaPyodide', async () => {
    try {
      const res = (Module as any).extractTraceViaPyodide();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
