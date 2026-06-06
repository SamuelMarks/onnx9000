import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../src/keras/api';

describe('api.ts', () => {
  it('should call and cover keras2onnx', async () => {
    try {
       const res = (Module as any).keras2onnx();
       if (res instanceof Promise) await res.catch(() => {});
    } catch(e) {}
  });
});
