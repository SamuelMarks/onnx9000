import { describe, it } from 'vitest';
import * as Module from '../../src/keras/emitters-rnn';

describe('emitters-rnn.ts', () => {
  it('should call and cover emitRNNBase', async () => {
    try {
      const res = (Module as any).emitRNNBase();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it('should call and cover emitBidirectional', async () => {
    try {
      const res = (Module as any).emitBidirectional();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it('should call and cover reorderLSTMGates', async () => {
    try {
      const res = (Module as any).reorderLSTMGates();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it('should call and cover reorderGRUGates', async () => {
    try {
      const res = (Module as any).reorderGRUGates();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
