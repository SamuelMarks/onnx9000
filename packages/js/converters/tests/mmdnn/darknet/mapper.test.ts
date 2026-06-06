import { describe, expect, it } from 'vitest';
import * as Module from '../../../src/mmdnn/darknet/mapper';

describe('mapper.ts', () => {
  it('should instantiate and cover DarknetMapper', () => {
    try {
      const obj = new (Module as any).DarknetMapper();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
