import { describe, it, expect } from 'vitest';
import { emitActivation, emitDense, emitIdentity } from '../src/keras/emitters.js';

describe('emitters', () => {
  it('emitActivation', () => {
    expect(emitActivation('relu', 'in', 'out', 'n')).toEqual([
      { opType: 'Relu', inputs: ['in'], outputs: ['out'], name: 'n', attributes: [] },
    ]);
    expect(emitActivation('softmax', 'in', 'out', 'n')).toEqual([
      {
        opType: 'Softmax',
        inputs: ['in'],
        outputs: ['out'],
        name: 'n',
        attributes: [{ name: 'axis', i: -1, type: 'INT' }],
      },
    ]);
    expect(emitActivation('sigmoid', 'in', 'out', 'n')).toEqual([
      { opType: 'Sigmoid', inputs: ['in'], outputs: ['out'], name: 'n', attributes: [] },
    ]);
    expect(emitActivation('tanh', 'in', 'out', 'n')).toEqual([
      { opType: 'Tanh', inputs: ['in'], outputs: ['out'], name: 'n', attributes: [] },
    ]);
    expect(emitActivation('softplus', 'in', 'out', 'n')).toEqual([
      { opType: 'Softplus', inputs: ['in'], outputs: ['out'], name: 'n', attributes: [] },
    ]);
    expect(emitActivation('softsign', 'in', 'out', 'n')).toEqual([
      { opType: 'Softsign', inputs: ['in'], outputs: ['out'], name: 'n', attributes: [] },
    ]);
    expect(emitActivation('linear', 'in', 'out', 'n')).toEqual([
      { opType: 'Identity', inputs: ['in'], outputs: ['out'], name: 'n', attributes: [] },
    ]);
    expect(emitActivation('elu', 'in', 'out', 'n', { alpha: 2.0 })).toEqual([
      {
        opType: 'Elu',
        inputs: ['in'],
        outputs: ['out'],
        name: 'n',
        attributes: [{ name: 'alpha', f: 2.0, type: 'FLOAT' }],
      },
    ]);
    expect(emitActivation('selu', 'in', 'out', 'n')).toEqual([
      { opType: 'Selu', inputs: ['in'], outputs: ['out'], name: 'n', attributes: [
        { name: 'alpha', f: 1.6732632423543772848170429916717, type: 'FLOAT' },
        { name: 'gamma', f: 1.0507009873554804934193349852946, type: 'FLOAT' }
      ] },
    ]);
    expect(emitActivation('leaky_relu', 'in', 'out', 'n', { alpha: 0.1 })).toEqual([
      {
        opType: 'LeakyRelu',
        inputs: ['in'],
        outputs: ['out'],
        name: 'n',
        attributes: [{ name: 'alpha', f: 0.1, type: 'FLOAT' }],
      },
    ]);
    expect(emitActivation('prelu', 'in', 'out', 'n', { alphaWeightName: 'aw' })).toEqual([
      { opType: 'PRelu', inputs: ['in', 'aw'], outputs: ['out'], name: 'n', attributes: [] },
    ]);
    expect(emitActivation('thresholded_relu', 'in', 'out', 'n', { theta: 0.5 })).toEqual([
      {
        opType: 'ThresholdedRelu',
        inputs: ['in'],
        outputs: ['out'],
        name: 'n',
        attributes: [{ name: 'alpha', f: 0.5, type: 'FLOAT' }],
      },
    ]);
    expect(emitActivation('gelu', 'in', 'out', 'n')).toEqual([
      { opType: 'Gelu', inputs: ['in'], outputs: ['out'], name: 'n', attributes: [
         { name: 'approximate', s: 'none', type: 'STRING' }
      ] },
    ]);
    expect(emitActivation('hard_sigmoid', 'in', 'out', 'n')).toEqual([
      {
        opType: 'HardSigmoid',
        inputs: ['in'],
        outputs: ['out'],
        name: 'n',
        attributes: [
          { name: 'alpha', f: 0.2, type: 'FLOAT' },
          { name: 'beta', f: 0.5, type: 'FLOAT' },
        ],
      },
    ]);
    expect(emitActivation('swish', 'in', 'out', 'n')).toHaveLength(2);
    expect(emitActivation('silu', 'in', 'out', 'n')).toHaveLength(2);

    expect(() => emitActivation('unknown', 'in', 'out', 'n')).toThrow(
      'Unsupported activation: unknown',
    );
  });

  it('emitDense', () => {
    const nodes = emitDense('in', 'out', 'w', 'b', 'relu', 'n');
    expect(nodes).toHaveLength(3); // MatMul, Add, Relu

    const linear = emitDense('in', 'out', 'w', undefined, 'linear', 'n');
    expect(linear).toHaveLength(1); // just MatMul
  });

  it('emitIdentity', () => {
    expect(emitIdentity('in', 'out', 'n')).toEqual([
      { opType: 'Identity', inputs: ['in'], outputs: ['out'], name: 'n', attributes: [] },
    ]);
  });
});
