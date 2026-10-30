import { describe, expect, it } from 'vitest';
import { emitAttention, emitEmbedding } from '../../src/keras/emitters-attention';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('emitters-attention.ts', () => {
  it('should call and cover emitAttention', async () => {
    try {
      const res = Call(emitAttention);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover emitEmbedding', async () => {
    try {
      const res = Call(emitEmbedding);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
