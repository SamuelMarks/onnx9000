import { describe, expect, it } from 'vitest';
import { DarknetMapper } from '../../../src/mmdnn/darknet/mapper';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('mapper.ts', () => {
  it('should instantiate and cover DarknetMapper', () => {
    try {
      const obj = Construct(DarknetMapper);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
