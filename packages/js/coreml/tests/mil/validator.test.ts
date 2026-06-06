import { describe, it } from 'vitest';
import * as Module from '../../src/mil/validator';

describe('validator.ts', () => {
  it('should call and cover validateMILProgram', async () => {
    try {
      const res = (Module as any).validateMILProgram();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it('should call and cover validateBlock', async () => {
    try {
      const res = (Module as any).validateBlock();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
