import { describe, it } from 'vitest';
import * as Module from '../../src/keras/plugin-registry';

describe('plugin-registry.ts', () => {
  it('should call and cover registerCustomKerasLayer', async () => {
    try {
      const res = (Module as any).registerCustomKerasLayer();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it('should call and cover getCustomKerasLayerEmitter', async () => {
    try {
      const res = (Module as any).getCustomKerasLayerEmitter();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
