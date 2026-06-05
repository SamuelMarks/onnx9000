import { describe, it, expect } from 'vitest';
import { PolyfillMLGraphBuilder } from '../src/builder.js';
import { PolyfillMLContext } from '../src/context.js';

describe('WebNN Builder', () => {
  it('should build operations', async () => {
    const ctx = new PolyfillMLContext();
    const b = new PolyfillMLGraphBuilder(ctx);

    const i1 = b.input('a', { dataType: 'float32', dimensions: [1] });
    const c1 = b.constant({ dataType: 'float32', dimensions: [1] }, new Float32Array([1]));

    const a1 = b.add(i1, c1);
    expect(a1).toBeDefined();

    const g = await b.build({ out: a1 });
    expect(g).toBeDefined();
  });
});
