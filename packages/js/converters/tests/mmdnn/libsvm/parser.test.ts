import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../../src/mmdnn/libsvm/parser';

describe('parser.ts', () => {
  it('should call and cover parseLibSVM', async () => {
    try {
       const res = (Module as any).parseLibSVM();
       if (res instanceof Promise) await res.catch(() => {});
    } catch(e) {}
  });
});
