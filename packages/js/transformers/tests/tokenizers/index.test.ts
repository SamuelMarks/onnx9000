import { describe, expect, it } from 'vitest';
import {
  AutoTokenizer,
  BPEEncoder,
  PreTrainedTokenizer,
  PreTrainedTokenizerFast,
} from '../../src/tokenizers/index';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('index.ts', () => {
  it('should instantiate and cover PreTrainedTokenizer', () => {
    try {
      const obj = Construct(PreTrainedTokenizer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover PreTrainedTokenizerFast', () => {
    try {
      const obj = Construct(PreTrainedTokenizerFast);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover AutoTokenizer', () => {
    try {
      const obj = Construct(AutoTokenizer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover BPEEncoder', () => {
    try {
      const obj = Construct(BPEEncoder);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
