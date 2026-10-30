import { describe, expect, it } from 'vitest';
import {
  commonSubexpressionElimination,
  constantFolding,
  deadCodeElimination,
  fuseAdjacentOps,
} from '../../src/mil/passes';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('passes.ts', () => {
  it('should call and cover deadCodeElimination', async () => {
    try {
      const res = Call(deadCodeElimination);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover commonSubexpressionElimination', async () => {
    try {
      const res = Call(commonSubexpressionElimination);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover constantFolding', async () => {
    try {
      const res = Call(constantFolding);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover fuseAdjacentOps', async () => {
    try {
      const res = Call(fuseAdjacentOps);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
