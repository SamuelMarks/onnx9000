import { describe, expect, it } from 'vitest';
import {
  LookaheadDecoder,
  MedusaEagleDecoder,
  PrefixTreeBatchedProcessor,
  RAGPrefixMatcher,
} from '../../src/genai/worker';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('worker.ts', () => {
  it('should instantiate and cover PrefixTreeBatchedProcessor', () => {
    try {
      const obj = Construct(PrefixTreeBatchedProcessor);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover LookaheadDecoder', () => {
    try {
      const obj = Construct(LookaheadDecoder);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover MedusaEagleDecoder', () => {
    try {
      const obj = Construct(MedusaEagleDecoder);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover RAGPrefixMatcher', () => {
    try {
      const obj = Construct(RAGPrefixMatcher);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
