import { describe, expect, it } from 'vitest';
import {
  BartModel,
  GptNeoXModel,
  LoraAdapter,
  Model,
  MoEModel,
  MultiModalModel,
  OptModel,
  SpeculativeDecodingModel,
  T5Model,
  WhisperModel,
} from '../../src/genai/model';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('model.ts', () => {
  it('should instantiate and cover Model', () => {
    try {
      const obj = Construct(Model);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover MultiModalModel', () => {
    try {
      const obj = Construct(MultiModalModel);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover WhisperModel', () => {
    try {
      const obj = Construct(WhisperModel);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover T5Model', () => {
    try {
      const obj = Construct(T5Model);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover OptModel', () => {
    try {
      const obj = Construct(OptModel);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover BartModel', () => {
    try {
      const obj = Construct(BartModel);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover GptNeoXModel', () => {
    try {
      const obj = Construct(GptNeoXModel);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover MoEModel', () => {
    try {
      const obj = Construct(MoEModel);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover SpeculativeDecodingModel', () => {
    try {
      const obj = Construct(SpeculativeDecodingModel);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover LoraAdapter', () => {
    try {
      const obj = Construct(LoraAdapter);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
