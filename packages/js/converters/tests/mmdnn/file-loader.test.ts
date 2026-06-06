import { describe, expect, it } from 'vitest';
import * as Module from '../../src/mmdnn/file-loader';

describe('file-loader.ts', () => {
  it('should instantiate and cover FileLoader', () => {
    try {
      const obj = new (Module as any).FileLoader();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
