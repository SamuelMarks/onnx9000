import { describe, it, expect } from 'vitest';
import { MBConv } from '../src/models/efficientnet.js';
import { Tensor } from '../src/index.js';

describe('Coverage Core 3', () => {
  it('MBConv depthwiseConv kernelSize is array', () => {
    const block = new MBConv(32, 64, 3, 1, 1, 'test');
    block.depthwiseConv.kernelSize = [3, 3]; // Array
    const x = new Tensor(
      'x',
      [1, 32, 224, 224],
      'float32',
      false,
      true,
      new Float32Array(1 * 32 * 224 * 224),
    );
    block.call(x);
  });
});
