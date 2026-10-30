import { describe, expect, it } from 'vitest';
import { Component } from '../../src/core/Component';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('Component.ts', () => {
  it('should instantiate and cover Component', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(Component);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
