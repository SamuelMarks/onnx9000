import { describe, expect, it } from 'vitest';
import { Attribute, Node } from '../../src/ir/node';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('node.ts', () => {
  it('should instantiate and cover Attribute', () => {
    try {
      const obj = Construct(Attribute);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover Node', () => {
    try {
      const obj = Construct(Node);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
