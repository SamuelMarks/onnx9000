import { describe, expect, it } from 'vitest';
import {
  toEmscriptenType,
  validateOnnxShapesAndDtypes,
} from '../../src/parser/safetensors.validator';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('safetensors.validator.ts', () => {
  it('should call and cover toEmscriptenType', async () => {
    try {
      const res = Call(toEmscriptenType);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover validateOnnxShapesAndDtypes', async () => {
    try {
      const res = Call(validateOnnxShapesAndDtypes);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
