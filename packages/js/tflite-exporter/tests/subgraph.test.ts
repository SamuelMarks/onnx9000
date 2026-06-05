import { describe, it, expect, vi } from 'vitest';
import { compileGraphToTFLite } from '../src/compiler/subgraph.js';
import { Graph } from '@onnx9000/core';

vi.mock('../src/compiler/layout.js', () => ({
  LayoutOptimizer: class {
    optimize() {}
  },
}));
vi.mock('../quantization/quantizer', () => ({
  Quantizer: class {
    quantize() {}
    getQuantizationOffset() {
      return 0;
    }
  },
}));
vi.mock('../optimizations/edgetpu', () => ({
  EdgeTPUOptimizer: class {
    optimize() {
      return [];
    }
  },
}));

describe('subgraph', () => {
  it('should compile', () => {
    const g = new Graph('test');
    const exp: any = {
      builder: {
        startVector: vi.fn(),
        addInt32: vi.fn(),
        endVector: vi.fn().mockReturnValue(0),
        createString: vi.fn(),
      },
      getOrAddOperatorCode: vi.fn().mockReturnValue(0),
    };

    const res = compileGraphToTFLite(g, exp);
    expect(res).toBeDefined();
  });
});
