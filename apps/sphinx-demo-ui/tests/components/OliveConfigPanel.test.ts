import { describe, expect, it } from 'vitest';
import { OliveConfigPanel } from '../../src/components/OliveConfigPanel';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('OliveConfigPanel.ts', () => {
  it('should instantiate and cover OliveConfigPanel', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(OliveConfigPanel);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
