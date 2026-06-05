import { describe, it, expect } from 'vitest';
import { PolyfillMLTensor } from '../src/tensor.js';

describe('WebNN Tensor', () => {
  it('should destroy', () => {
    const t = new PolyfillMLTensor({ dataType: 'float32', dimensions: [10] });
    expect(t.internalBuffer).toBeDefined();
    t.destroy();
    expect(t.internalBuffer).toBeNull();
  });
});
