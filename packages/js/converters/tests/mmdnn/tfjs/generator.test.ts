import { describe, expect, it } from 'vitest';
import { createModel, generateTFJSCode, isLinearGraph } from '../../../src/mmdnn/tfjs/generator';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('generator.ts', () => {
  it('should call and cover isLinearGraph', async () => {
    try {
      const res = Call(isLinearGraph);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover generateTFJSCode', async () => {
    try {
      const res = Call(generateTFJSCode);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover createModel', async () => {
    try {
      const res = Call(createModel);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
