import { Graph, Tensor } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { InitializerInspector } from '../src/components/initializers/inspector.js';
import { GraphMutator } from '../src/GraphMutator.js';

describe('InitializerInspector', () => {
  it('should render scalar', () => {
    const container = document.createElement('div');
    const mutator = new GraphMutator(new Graph('test'));
    const insp = new InitializerInspector(container, mutator);
    const tensor = new Tensor('t', [1], 'float32', true, false, new Uint8Array(4));
    insp.render(tensor);
    expect(container.innerHTML).toContain('Scalar Editor');
  });
});
