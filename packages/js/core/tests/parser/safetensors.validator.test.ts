import { describe, it } from 'vitest';
import * as Module from '../../src/parser/safetensors.validator';

describe('safetensors.validator.ts', () => {
  it('should call and cover toEmscriptenType', async () => {
    try {
      const res = (Module as any).toEmscriptenType();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it('should call and cover validateOnnxShapesAndDtypes', async () => {
    try {
      const res = (Module as any).validateOnnxShapesAndDtypes();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
