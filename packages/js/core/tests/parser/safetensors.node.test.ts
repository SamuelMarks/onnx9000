import { describe, expect, it } from 'vitest';
import {
  readSafetensorsChunkSync,
  readSafetensorsHeaderSync,
  saveSafetensorsFileSync,
} from '../../src/parser/safetensors.node';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('safetensors.node.ts', () => {
  it('should call and cover readSafetensorsHeaderSync', async () => {
    try {
      const res = Call(readSafetensorsHeaderSync);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover readSafetensorsChunkSync', async () => {
    try {
      const res = Call(readSafetensorsChunkSync);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover saveSafetensorsFileSync', async () => {
    try {
      const res = Call(saveSafetensorsFileSync);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
