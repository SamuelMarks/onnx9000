import { describe, it, expect } from 'vitest';
import { registry } from '../src/index';

describe('CustomOps Registry', () => {
  it('should register and retrieve an op', () => {
    const myOp = () => 'ok';
    registry.register('MyOp', myOp);
    expect(registry.listOps()).toContain('MyOp');
    expect(registry.getOp('MyOp')?.()).toBe('ok');
  });
});
