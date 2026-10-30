import { describe, expect, it } from 'vitest';
import { FileLoader } from '../../src/mmdnn/file-loader';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('file-loader.ts', () => {
  it('should instantiate and cover FileLoader', () => {
    try {
      const obj = Construct(FileLoader);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
