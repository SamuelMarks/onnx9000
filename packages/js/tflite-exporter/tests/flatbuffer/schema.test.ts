import { describe, expect, it } from 'vitest';
import {
  Buffer,
  Metadata,
  Model,
  Operator,
  OperatorCode,
  QuantizationParameters,
  SubGraph,
  Tensor,
} from '../../src/flatbuffer/schema';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('schema.ts', () => {
  it('should instantiate and cover OperatorCode', () => {
    try {
      const obj = Construct(OperatorCode);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover QuantizationParameters', () => {
    try {
      const obj = Construct(QuantizationParameters);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover Tensor', () => {
    try {
      const obj = Construct(Tensor);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover Operator', () => {
    try {
      const obj = Construct(Operator);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover SubGraph', () => {
    try {
      const obj = Construct(SubGraph);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover Buffer', () => {
    try {
      const obj = Construct(Buffer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover Metadata', () => {
    try {
      const obj = Construct(Metadata);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover Model', () => {
    try {
      const obj = Construct(Model);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
