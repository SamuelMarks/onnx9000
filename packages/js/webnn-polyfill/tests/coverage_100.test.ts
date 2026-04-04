import { describe, it, expect } from 'vitest';
import { PolyfillMLGraphBuilder } from '../src/builder';
import { PolyfillMLContext } from '../src/context';
import { PolyfillMLOperand } from '../src/operand';

describe('Coverage 100 WebNN Polyfill', () => {
  it('builder build sync names', async () => {
    const b = new PolyfillMLGraphBuilder(new PolyfillMLContext());
    const input = b.input('input', { dataType: 'float32', dimensions: [1] });

    // Let's create an operand with name identical to output key
    const op = new PolyfillMLOperand('same_name', 'float32', [1]);
    (op as any).sourceNode = {
      name: 'same_name',
      inputs: [],
      outputs: ['same_name'],
      opType: 'Identity',
    } as any;

    // Normally build takes a dict of outputs.
    const graph = await b.build({ same_name: op });
    expect(graph).toBeDefined();
  });
});
