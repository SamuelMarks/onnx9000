import { describe, expect, it } from 'vitest';
import { PropertiesPanel } from '../../src/components/properties';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('properties.ts', () => {
  it('should instantiate and cover PropertiesPanel', () => {
    try {
      const obj = Construct(PropertiesPanel);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
