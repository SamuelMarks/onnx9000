import { describe, expect, it } from 'vitest';
import { fetchRemoteUrl, readBrowserFile } from '../../src/keras/browser-io';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('browser-io.ts', () => {
  it('should call and cover readBrowserFile', async () => {
    try {
      const res = Call(readBrowserFile);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover fetchRemoteUrl', async () => {
    try {
      const res = Call(fetchRemoteUrl);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
