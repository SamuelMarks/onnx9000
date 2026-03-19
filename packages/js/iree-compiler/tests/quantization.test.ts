import { describe, it, expect } from 'vitest';
import { Region, Operation, Block } from '../src/ir/core.js';
import { QuantizationOptimizer } from '../src/passes/quantization.js';

describe('Quantization Passes', () => {
  it('should run quantization lowering', () => {
    const region = new Region();
    const block = new Block(region);
    region.pushBlock(block);

    block.pushOperation(new Operation('web.mhlo.dynamic_quantize_linear', [], [], {}));

    const optimizer = new QuantizationOptimizer();
    expect(() => optimizer.runAll(region)).not.toThrow();

    const wgsl = optimizer.emitW4A16WGSL('// kernel');
    expect(wgsl).toContain('unpack_w4');
    expect(wgsl).toContain('val >> shift');
  });

  it('should validate size tracking', () => {
    const optimizer = new QuantizationOptimizer();
    expect(() => optimizer.trackQuantizationSize(100, 50)).not.toThrow();
  });
});
