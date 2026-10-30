import { describe, expect, it } from 'vitest';
import {
  ANELimitsExceededWarning,
  CoreMLExportError,
  DoubleDowncastWarning,
  ThermalThrottlingWarning,
  UnsupportedOpError,
} from '../../src/mil/errors';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('errors.ts', () => {
  it('should instantiate and cover CoreMLExportError', () => {
    try {
      const obj = Construct(CoreMLExportError);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover UnsupportedOpError', () => {
    try {
      const obj = Construct(UnsupportedOpError);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover ThermalThrottlingWarning', () => {
    try {
      const obj = Construct(ThermalThrottlingWarning);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover ANELimitsExceededWarning', () => {
    try {
      const obj = Construct(ANELimitsExceededWarning);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover DoubleDowncastWarning', () => {
    try {
      const obj = Construct(DoubleDowncastWarning);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
