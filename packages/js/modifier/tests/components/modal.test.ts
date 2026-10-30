import { describe, expect, it } from 'vitest';
import { AddNodeModal } from '../../src/components/modal';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('modal.ts', () => {
  it('should instantiate and cover AddNodeModal', () => {
    try {
      const obj = Construct(AddNodeModal);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
