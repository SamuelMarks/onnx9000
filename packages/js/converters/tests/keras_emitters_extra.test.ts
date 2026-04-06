import { describe, it, expect, vi } from 'vitest';
import { emitDense, emitActivation, emitIdentity } from '../src/keras/emitters.js';
import {
  emitGroupNormalization,
  emitBatchNormalization,
  emitLayerNormalization,
} from '../src/keras/emitters-norm.js';
import { emitMerge } from '../src/keras/emitters-merge.js';

describe('Keras Emitters Exhaustive Coverage', () => {
  const builder = {
    addNode: vi.fn(),
    addTensor: vi.fn(),
    graph: { tensors: {}, initializers: [] },
  };

  it('should cover normalization emitters', () => {
    emitGroupNormalization(builder as Object, 'gn', ['in'], { groups: 32 });
    emitBatchNormalization(builder as Object, 'bn', ['in'], { epsilon: 1e-3 });
    emitLayerNormalization(builder as Object, 'ln', ['in'], { axis: -1 });
  });

  it('should cover merge emitters', () => {
    const inputs = ['in1', 'in2'];
    emitMerge(builder as Object, 'add', inputs, { mode: 'sum' });
    emitMerge(builder as Object, 'mul', inputs, { mode: 'mul' });
    emitMerge(builder as Object, 'sub', inputs, { mode: 'sub' });
    emitMerge(builder as Object, 'div', inputs, { mode: 'div' });
    emitMerge(builder as Object, 'ave', inputs, { mode: 'ave' });
    emitMerge(builder as Object, 'max', inputs, { mode: 'max' });
    emitMerge(builder as Object, 'min', inputs, { mode: 'min' });
  });

  it('should cover base emitters and edge cases', () => {
    emitDense(builder as Object, 'dense', ['in'], { units: 10 });

    const activations = [
      'relu',
      'sigmoid',
      'tanh',
      'softmax',
      'softplus',
      'softsign',
      'elu',
      'selu',
      'gelu',
      'hard_sigmoid',
      'linear',
      'mish',
      'leaky_relu',
      'thresholded_relu',
      'swish',
      'silu',
      'hard_swish',
    ];
    for (const act of activations) {
      emitActivation(act, 'in', 'out', 'node_' + act, {
        alpha: 0.1,
        theta: 1.0,
        alphaWeightName: 'alpha',
      });
    }

    emitIdentity(builder as Object, 'id', ['in'], {});
    emitDense(builder as Object, 'dense_no_cfg', ['in'], {} as Object);
  });
});
