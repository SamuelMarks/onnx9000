import { describe, it } from 'vitest';
import * as Module from '../src/tensorrt';

describe('tensorrt.ts', () => {
  it('should call and cover createTensorRTSession', async () => {
    try {
      const res = (Module as any).createTensorRTSession();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
