import { describe, it, expect } from 'vitest';
import { Region, Operation, Block } from '../src/ir/core.js';
import { Optimizer } from '../src/passes/optimization.js';

describe('MLIR Optimization Passes', () => {
  it('should run all optimizations', () => {
    const region = new Region();
    const block = new Block(region);
    region.pushBlock(block);

    block.pushOperation(new Operation('web.vm.add.i32', [], [], {}));
    block.pushOperation(new Operation('web.mhlo.convolution', [], [], {}));

    const optimizer = new Optimizer();
    expect(() => optimizer.runAll(region)).not.toThrow();

    // Ensure DCE or other passes didn't indiscriminately crash or clear everything
    // For the dummy, operations remain
    expect(block.operations.length).toBeGreaterThanOrEqual(2);
  });
});

it('should optimize attention', () => {
  const region = new Region();
  const block = new Block(region);
  region.pushBlock(block);
  block.pushOperation(new Operation('web.linalg.matmul', [], [], {}));

  const optimizer = new Optimizer();
  optimizer.optimizeAttentionPatterns(region);
});
