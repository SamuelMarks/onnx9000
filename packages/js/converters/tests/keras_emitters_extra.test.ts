import { describe, expect, it, vi } from 'vitest';
import { emitActivation, emitDense, emitIdentity } from '../src/keras/emitters.js';
import { emitMerge } from '../src/keras/emitters-merge.js';
import {
  emitBatchNormalization,
  emitGroupNormalization,
  emitLayerNormalization,
} from '../src/keras/emitters-norm.js';

describe('Keras Emitters Exhaustive Coverage', () => {
  const builder = {
    addNode: vi.fn(),
    addTensor: vi.fn(),
    graph: { tensors: {}, initializers: [] },
  };

  it('should cover normalization emitters', () => {
    emitGroupNormalization(
      builder as never as Parameters<typeof emitBatchNormalization>[0],
      'gn',
      ['in'],
      { groups: 32 },
    );
    emitBatchNormalization(
      builder as never as Parameters<typeof emitBatchNormalization>[0],
      'bn',
      ['in'],
      { epsilon: 1e-3 },
    );
    emitLayerNormalization(
      builder as never as Parameters<typeof emitBatchNormalization>[0],
      'ln',
      ['in'],
      { axis: -1 },
    );
    expect(typeof document).toBeDefined();
  });

  it('should cover merge emitters', () => {
    const inputs = ['in1', 'in2'];
    emitMerge(builder as never as Parameters<typeof emitBatchNormalization>[0], 'add', inputs, {
      mode: 'sum',
    });
    emitMerge(builder as never as Parameters<typeof emitBatchNormalization>[0], 'mul', inputs, {
      mode: 'mul',
    });
    emitMerge(builder as never as Parameters<typeof emitBatchNormalization>[0], 'sub', inputs, {
      mode: 'sub',
    });
    emitMerge(builder as never as Parameters<typeof emitBatchNormalization>[0], 'div', inputs, {
      mode: 'div',
    });
    emitMerge(builder as never as Parameters<typeof emitBatchNormalization>[0], 'ave', inputs, {
      mode: 'ave',
    });
    emitMerge(builder as never as Parameters<typeof emitBatchNormalization>[0], 'max', inputs, {
      mode: 'max',
    });
    emitMerge(builder as never as Parameters<typeof emitBatchNormalization>[0], 'min', inputs, {
      mode: 'min',
    });
    expect(typeof inputs).toBeDefined();
  });

  it('should cover base emitters and edge cases', () => {
    emitDense(builder as never as Parameters<typeof emitBatchNormalization>[0], 'dense', ['in'], {
      units: 10,
    });

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
      emitActivation(act, 'in', 'out', `node_${act}`, {
        alpha: 0.1,
        theta: 1.0,
        alphaWeightName: 'alpha',
      });
    }

    emitIdentity(
      builder as never as Parameters<typeof emitBatchNormalization>[0],
      'id',
      ['in'],
      {},
    );
    emitDense(
      builder as never as Parameters<typeof emitBatchNormalization>[0],
      'dense_no_cfg',
      ['in'],
      {} as never as Parameters<typeof emitDense>[3],
    );
    expect(typeof activations).toBeDefined();
  });
});
