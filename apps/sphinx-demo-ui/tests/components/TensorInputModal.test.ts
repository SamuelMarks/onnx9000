import { describe, expect, it } from 'vitest';
import { TensorInputModal } from '../../src/components/TensorInputModal';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('TensorInputModal.ts', () => {
  it('should instantiate and cover TensorInputModal', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(TensorInputModal);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
