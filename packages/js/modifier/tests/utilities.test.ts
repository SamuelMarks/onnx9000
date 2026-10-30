import { Graph, ValueInfo } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { ModifierUtilities } from '../src/components/utilities.js';
import { GraphMutator } from '../src/GraphMutator.js';

describe('ModifierUtilities', () => {
  it('should change batch size', () => {
    const g = new Graph('test');
    g.inputs.push(new ValueInfo('in', [1, 2], 'float32'));
    const mutator = new GraphMutator(g);
    const utils = new ModifierUtilities(mutator);
    utils.changeBatchSize(8);
    expect(mutator.graph.inputs[0].shape[0]).toBe(8);
  });
});
