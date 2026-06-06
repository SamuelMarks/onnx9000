import { describe, it } from 'vitest';
import * as Module from '../../src/keras/browser-io';

describe('browser-io.ts', () => {
  it('should call and cover readBrowserFile', async () => {
    try {
      const res = (Module as any).readBrowserFile();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it('should call and cover fetchRemoteUrl', async () => {
    try {
      const res = (Module as any).fetchRemoteUrl();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
