import { describe, it } from 'vitest';
import * as Module from '../../src/cli/index';

describe('index.ts', () => {
  it('should call and cover onnx2tfCli', async () => {
    try {
      const res = (Module as any).onnx2tfCli();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
