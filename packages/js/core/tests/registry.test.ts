import { describe, it, expect } from 'vitest';
import { globalRegistry, register_op } from '../src/ops/registry.js';

describe('registry', () => {
  it('should register and get', () => {
    class MockOp {
      execute() {
        return [];
      }
    }
    register_op('test', 'MockOp')(MockOp);

    const impl = globalRegistry.get_op('test', 'MockOp');
    expect(impl).toBe(MockOp);
  });
});
