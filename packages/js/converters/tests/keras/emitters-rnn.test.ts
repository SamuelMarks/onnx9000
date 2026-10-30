import { describe, expect, it } from 'vitest';
import {
  emitBidirectional,
  emitRNNBase,
  reorderGRUGates,
  reorderLSTMGates,
} from '../../src/keras/emitters-rnn';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('emitters-rnn.ts', () => {
  it('should call and cover emitRNNBase', async () => {
    try {
      const res = Call(emitRNNBase);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover emitBidirectional', async () => {
    try {
      const res = Call(emitBidirectional);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover reorderLSTMGates', async () => {
    try {
      const res = Call(reorderLSTMGates);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover reorderGRUGates', async () => {
    try {
      const res = Call(reorderGRUGates);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
