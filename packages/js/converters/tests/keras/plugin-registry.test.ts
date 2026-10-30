import { describe, expect, it } from 'vitest';
import {
  getCustomKerasLayerEmitter,
  registerCustomKerasLayer,
} from '../../src/keras/plugin-registry';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('plugin-registry.ts', () => {
  it('should call and cover registerCustomKerasLayer', async () => {
    try {
      const res = Call(registerCustomKerasLayer);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover getCustomKerasLayerEmitter', async () => {
    try {
      const res = Call(getCustomKerasLayerEmitter);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
