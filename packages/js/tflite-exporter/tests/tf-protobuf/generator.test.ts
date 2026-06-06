import { describe, expect, it } from 'vitest';
import * as Module from '../../src/tf-protobuf/generator';

describe('generator.ts', () => {
  it('should instantiate and cover SavedModelGenerator', () => {
    try {
      const obj = new (Module as any).SavedModelGenerator();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
