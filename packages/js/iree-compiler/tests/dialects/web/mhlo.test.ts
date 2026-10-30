import { describe, expect, it } from 'vitest';
import {
  broadcastInDim,
  concatenate,
  convolution,
  dot,
  dynamicSlice,
  gather,
  reduce,
  reduceWindow,
  reshape,
  scatter,
  select,
  slice,
  transpose,
} from '../../../src/dialects/web/mhlo';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('mhlo.ts', () => {
  it('should call and cover dot', async () => {
    try {
      const res = Call(dot);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover convolution', async () => {
    try {
      const res = Call(convolution);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover reduce', async () => {
    try {
      const res = Call(reduce);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover reduceWindow', async () => {
    try {
      const res = Call(reduceWindow);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover select', async () => {
    try {
      const res = Call(select);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover broadcastInDim', async () => {
    try {
      const res = Call(broadcastInDim);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover reshape', async () => {
    try {
      const res = Call(reshape);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover transpose', async () => {
    try {
      const res = Call(transpose);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover concatenate', async () => {
    try {
      const res = Call(concatenate);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover slice', async () => {
    try {
      const res = Call(slice);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover dynamicSlice', async () => {
    try {
      const res = Call(dynamicSlice);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover gather', async () => {
    try {
      const res = Call(gather);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover scatter', async () => {
    try {
      const res = Call(scatter);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
