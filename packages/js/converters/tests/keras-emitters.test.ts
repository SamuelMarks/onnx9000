import { describe, expect, test } from 'vitest';
import { emitConv, emitSeparableConv } from '../src/keras/emitters-conv.js';
import { emitPool, emitGlobalPool } from '../src/keras/emitters-pool.js';
import {
  emitBatchNormalization,
  emitLayerNormalization,
  emitUnitNormalization,
  emitGroupNormalization,
  emitReshape,
  emitFlatten,
  emitTranspose,
  emitPad,
} from '../src/keras/emitters-norm.js';
import { emitMerge, emitConcat, emitDot } from '../src/keras/emitters-merge.js';
import { emitAttention, emitEmbedding } from '../src/keras/emitters-attention.js';
import { mapTfjsOpToOnnx } from '../src/keras/emitters-tfjs.js';
import {
  emitRNNBase,
  emitBidirectional,
  reorderLSTMGates,
  reorderGRUGates,
} from '../src/keras/emitters-rnn.js';

describe('emitters-conv', () => {
  test('emitConv with valid padding and strides', () => {
    const nodes = emitConv('Conv', 'in', 'out', 'w', 'b', 'conv1', {
      activation: 'relu',
      strides: [1, 1],
      dilations: [1, 1],
      padding: 'valid',
      kernelShape: [3, 3],
    });
    expect(nodes.length).toBe(2);
    expect(nodes[0].opType).toBe('Conv');
    expect(nodes[1].opType).toBe('Relu');
  });

  test('emitConv with same padding and inputShape', () => {
    const nodes = emitConv('Conv', 'in', 'out', 'w', undefined, 'conv1', {
      activation: 'linear',
      strides: [2, 2],
      dilations: [1, 1],
      padding: 'same',
      kernelShape: [3, 3],
      inputShape: [1, 3, 28, 28],
      groups: 2,
    });
    expect(nodes.length).toBe(1);
    expect(nodes[0].opType).toBe('Conv');
    const pads = nodes[0].attributes?.find((a) => a.name === 'pads');
    expect(pads).toBeDefined();
  });

  test('emitConv with same padding and NO inputShape', () => {
    const nodes = emitConv('Conv', 'in', 'out', 'w', undefined, 'conv1', {
      activation: 'linear',
      strides: [2, 2],
      dilations: [1, 1],
      padding: 'same',
      kernelShape: [3, 3],
      groups: 2,
    });
    expect(nodes.length).toBe(1);
    expect(nodes[0].opType).toBe('Conv');
    const auto_pad = nodes[0].attributes?.find((a) => a.name === 'auto_pad');
    expect(auto_pad?.s).toBe('SAME_UPPER');
  });

  test('emitSeparableConv', () => {
    const nodes = emitSeparableConv(
      'in',
      'out',
      'dw',
      'pw',
      'b',
      'sep_conv',
      {
        activation: 'relu',
        strides: [1, 1],
        dilations: [1, 1],
        padding: 'valid',
        kernelShape: [3, 3],
      },
      32,
    );
    expect(nodes.length).toBe(3); // depthwise (linear) -> pointwise -> relu
    expect(nodes[0].opType).toBe('Conv');
    expect(nodes[1].opType).toBe('Conv');
    expect(nodes[2].opType).toBe('Relu');
  });
});

describe('emitters-pool', () => {
  test('emitPool Max/Average', () => {
    const nodesMax = emitPool('Max', 'in', 'out', 'pool1', {
      poolSize: [2, 2],
      strides: [2, 2],
      padding: 'valid',
    });
    expect(nodesMax[0].opType).toBe('MaxPool');
    expect(nodesMax[0].attributes?.find((a) => a.name === 'auto_pad')?.s).toBe('VALID');

    const nodesAvg = emitPool('Average', 'in', 'out', 'pool2', {
      poolSize: [2, 2],
      strides: [2, 2],
      padding: 'same',
    });
    expect(nodesAvg[0].opType).toBe('AveragePool');
    expect(nodesAvg[0].attributes?.find((a) => a.name === 'auto_pad')?.s).toBe('SAME_UPPER');
  });

  test('emitGlobalPool', () => {
    const nodesKeep = emitGlobalPool('Max', 'in', 'out', 'gpool1', { keepDims: true });
    expect(nodesKeep.length).toBe(1);
    expect(nodesKeep[0].opType).toBe('GlobalMaxPool');

    const nodesSqueeze = emitGlobalPool('Average', 'in', 'out', 'gpool2', { keepDims: false });
    expect(nodesSqueeze.length).toBe(2);
    expect(nodesSqueeze[0].opType).toBe('GlobalAveragePool');
    expect(nodesSqueeze[1].opType).toBe('Squeeze');
  });
});

describe('emitters-norm', () => {
  test('emitBatchNormalization', () => {
    const nodes = emitBatchNormalization(
      'in',
      'out',
      'gamma',
      'beta',
      'mean',
      'var',
      1e-5,
      0.9,
      'bn',
    );
    expect(nodes.length).toBe(1);
    expect(nodes[0].opType).toBe('BatchNormalization');
    expect(nodes[0].inputs).toEqual(['in', 'gamma', 'beta', 'mean', 'var']);
  });

  test('emitLayerNormalization', () => {
    const nodes = emitLayerNormalization('in', 'out', 'gamma', 'beta', -1, 1e-5, 'ln');
    expect(nodes[0].opType).toBe('LayerNormalization');

    const nodesNoWeights = emitLayerNormalization(
      'in',
      'out',
      undefined,
      undefined,
      -1,
      1e-5,
      'ln2',
    );
    expect(nodesNoWeights[0].inputs).toEqual(['in']);
  });

  test('emitReshape, emitFlatten, emitTranspose, emitPad', () => {
    expect(emitReshape('in', 'shape', 'out', 'reshape')[0].opType).toBe('Reshape');
    expect(emitFlatten('in', 'out', 1, 'flatten')[0].opType).toBe('Flatten');
    expect(emitTranspose('in', 'out', [0, 2, 1], 'transpose')[0].opType).toBe('Transpose');

    const pad1 = emitPad('in', 'out', 'pads', 'val', 'constant', 'pad1');
    expect(pad1[0].opType).toBe('Pad');
    expect(pad1[0].inputs.length).toBe(3);

    const pad2 = emitPad('in', 'out', 'pads', undefined, 'reflect', 'pad2');
    expect(pad2[0].inputs.length).toBe(2);
  });

  test('emitUnitNormalization', () => {
    const nodes = emitUnitNormalization('in', 'out', -1, 'unit_norm');
    expect(nodes[0].opType).toBe('LpNormalization');
    expect(nodes[0].attributes?.find((a) => a.name === 'p')?.i).toBe(2);
  });

  test('emitGroupNormalization', () => {
    const nodes = emitGroupNormalization('in', 'out', 32, 'gamma', 'beta', 1e-5, 'gn');
    expect(nodes[0].opType).toBe('GroupNormalization');
    expect(nodes[0].inputs).toEqual(['in', 'gamma', 'beta']);
    expect(nodes[0].attributes?.find((a) => a.name === 'num_groups')?.i).toBe(32);

    const nodesNoWeights = emitGroupNormalization(
      'in',
      'out',
      16,
      undefined,
      undefined,
      1e-5,
      'gn2',
    );
    expect(nodesNoWeights[0].inputs).toEqual(['in', '', '']);
  });
});

describe('emitters-merge', () => {
  test('emitMerge 2 inputs', () => {
    const nodes = emitMerge('Add', ['in1', 'in2'], 'out', 'add');
    expect(nodes.length).toBe(1);
    expect(nodes[0].opType).toBe('Add');
  });

  test('emitMerge multi inputs (Add)', () => {
    const nodes = emitMerge('Add', ['in1', 'in2', 'in3'], 'out', 'add');
    expect(nodes.length).toBe(2);
    expect(nodes[0].opType).toBe('Add');
    expect(nodes[1].opType).toBe('Add');
  });

  test('emitMerge multi inputs (Mean, Max, Min)', () => {
    const nodes = emitMerge('Mean', ['in1', 'in2', 'in3'], 'out', 'mean');
    expect(nodes.length).toBe(1);
    expect(nodes[0].opType).toBe('Mean');
    expect(nodes[0].inputs).toEqual(['in1', 'in2', 'in3']);
  });

  test('emitMerge < 2 inputs throws', () => {
    expect(() => emitMerge('Add', ['in1'], 'out', 'add')).toThrow();
  });

  test('emitConcat & emitDot', () => {
    expect(emitConcat(['in1', 'in2'], 'out', 1, 'concat')[0].opType).toBe('Concat');
    expect(emitDot('in1', 'in2', 'out', 1, 'dot')[0].opType).toBe('MatMul');
  });
});

describe('emitters-attention', () => {
  test('emitAttention', () => {
    const nodes1 = emitAttention('q', 'v', 'k', 'out', 'att1', false);
    expect(nodes1.length).toBe(4);

    const nodes2 = emitAttention('q', 'v', undefined, 'out', 'att2', true);
    expect(nodes2.length).toBe(5); // Add mask
  });

  test('emitEmbedding', () => {
    const nodes1 = emitEmbedding('in', 'w', 'out', 'emb', false);
    expect(nodes1.length).toBe(1);

    const nodes2 = emitEmbedding('in', 'w', 'out', 'emb', true);
    expect(nodes2.length).toBe(3);
  });
});

describe('emitters-tfjs', () => {
  test('mapTfjsOpToOnnx valid ops', () => {
    expect(mapTfjsOpToOnnx('Add', ['a', 'b'], 'out', 'add')[0].opType).toBe('Add');
    expect(mapTfjsOpToOnnx('AddV2', ['a', 'b'], 'out', 'add')[0].opType).toBe('Add');
    expect(mapTfjsOpToOnnx('Sub', ['a', 'b'], 'out', 'sub')[0].opType).toBe('Sub');
    expect(mapTfjsOpToOnnx('Mul', ['a', 'b'], 'out', 'mul')[0].opType).toBe('Mul');
    expect(mapTfjsOpToOnnx('RealDiv', ['a', 'b'], 'out', 'div')[0].opType).toBe('Div');
    expect(mapTfjsOpToOnnx('Div', ['a', 'b'], 'out', 'div2')[0].opType).toBe('Div');
    expect(mapTfjsOpToOnnx('MatMul', ['a', 'b'], 'out', 'mm')[0].opType).toBe('MatMul');
    expect(mapTfjsOpToOnnx('Square', ['a'], 'out', 'sq')[0].opType).toBe('Pow');
    expect(mapTfjsOpToOnnx('Sqrt', ['a'], 'out', 'sqrt')[0].opType).toBe('Sqrt');
    expect(mapTfjsOpToOnnx('Exp', ['a'], 'out', 'e')[0].opType).toBe('Exp');
    expect(mapTfjsOpToOnnx('Log', ['a'], 'out', 'l')[0].opType).toBe('Log');
    expect(mapTfjsOpToOnnx('Maximum', ['a', 'b'], 'out', 'max')[0].opType).toBe('Max');
    expect(mapTfjsOpToOnnx('Minimum', ['a', 'b'], 'out', 'min')[0].opType).toBe('Min');
    expect(mapTfjsOpToOnnx('Sum', ['a'], 'out', 's')[0].opType).toBe('ReduceSum');
    expect(mapTfjsOpToOnnx('Mean', ['a'], 'out', 'm')[0].opType).toBe('ReduceMean');
    expect(mapTfjsOpToOnnx('Max', ['a'], 'out', 'ma')[0].opType).toBe('ReduceMax');
    expect(mapTfjsOpToOnnx('Min', ['a'], 'out', 'mi')[0].opType).toBe('ReduceMin');
    expect(mapTfjsOpToOnnx('ArgMax', ['a'], 'out', 'am')[0].opType).toBe('ArgMax');
    expect(mapTfjsOpToOnnx('ArgMin', ['a'], 'out', 'ami')[0].opType).toBe('ArgMin');
    expect(mapTfjsOpToOnnx('Split', ['a', 'b'], 'out', 'sp')[0].opType).toBe('Split');
    expect(mapTfjsOpToOnnx('SplitV', ['a', 'b', 'c'], 'out', 'spv')[0].opType).toBe('Split');
    expect(mapTfjsOpToOnnx('Concat', ['a', 'b'], 'out', 'c')[0].opType).toBe('Concat');
    expect(mapTfjsOpToOnnx('ConcatV2', ['a', 'b', 'c'], 'out', 'cv2')[0].opType).toBe('Concat');
    expect(mapTfjsOpToOnnx('Slice', ['a', 'b', 'c'], 'out', 'sl')[0].opType).toBe('Slice');
    expect(mapTfjsOpToOnnx('StridedSlice', ['a', 'b', 'c', 'd'], 'out', 'ssl')[0].opType).toBe(
      'Slice',
    );
    expect(mapTfjsOpToOnnx('Gather', ['a', 'b'], 'out', 'g')[0].opType).toBe('Gather');
    expect(mapTfjsOpToOnnx('GatherV2', ['a', 'b', 'c'], 'out', 'gv2')[0].opType).toBe('Gather');
    expect(mapTfjsOpToOnnx('GatherNd', ['a', 'b'], 'out', 'gnd')[0].opType).toBe('GatherND');
    expect(mapTfjsOpToOnnx('Where', ['a', 'b', 'c'], 'out', 'w')[0].opType).toBe('Where');
    expect(mapTfjsOpToOnnx('TensorScatterUpdate', ['a', 'b', 'c'], 'out', 'tsu')[0].opType).toBe(
      'ScatterND',
    );
    expect(mapTfjsOpToOnnx('ResizeBilinear', ['a', 'b'], 'out', 'rb')[0].opType).toBe('Resize');
    expect(mapTfjsOpToOnnx('ResizeNearestNeighbor', ['a', 'b'], 'out', 'rnn')[0].opType).toBe(
      'Resize',
    );
  });

  test('mapTfjsOpToOnnx invalid op throws', () => {
    expect(() => mapTfjsOpToOnnx('InvalidOp', [], 'out', 'err')).toThrow();
  });
});

describe('emitters-rnn', () => {
  test('emitRNNBase', () => {
    const nodesSeq = emitRNNBase('LSTM', 'in', 'out', 'w', 'r', 'b', ['h0', 'c0'], 'lstm1', {
      returnSequences: true,
      returnState: false,
      goBackwards: false,
      stateful: false,
      linearBeforeReset: 1,
    });
    expect(nodesSeq[0].opType).toBe('LSTM');
    expect(nodesSeq[0].inputs).toEqual(['in', 'w', 'r', 'b', '', 'h0', 'c0']);
    expect(nodesSeq.length).toBe(3); // LSTM, Squeeze, Identity

    const nodesState = emitRNNBase('GRU', 'in', 'out', 'w', 'r', undefined, [], 'gru1', {
      returnSequences: false,
      returnState: true,
      goBackwards: true,
      stateful: false,
    });
    expect(nodesState[0].opType).toBe('GRU');
    expect(nodesState[0].inputs).toEqual(['in', 'w', 'r', '']);
    expect(nodesState[0].attributes?.find((a) => a.name === 'direction')?.s).toBe('reverse');
    expect(nodesState.length).toBe(4); // GRU, Squeeze, Squeeze, Identity

    // hit the `bName || ''` branch
    const nodesStateWithInitial = emitRNNBase(
      'GRU',
      'in',
      'out',
      'w',
      'r',
      undefined,
      ['h0'],
      'gru2',
      {
        returnSequences: false,
        returnState: true,
        goBackwards: false,
        stateful: false,
      },
    );
    expect(nodesStateWithInitial[0].inputs).toEqual(['in', 'w', 'r', '', '', 'h0']);
  });

  test('emitRNNBase with unrolling', () => {
    const nodes = emitRNNBase('RNN', 'in', 'out', 'w', 'r', 'b', [], 'rnn_unroll', {
      returnSequences: true,
      returnState: false,
      goBackwards: false,
      stateful: false,
      unroll: true,
      timeSteps: 2,
    });
    // For 2 timesteps: 2x(Gather, MatMul, MatMul, Add, Add, Tanh) + Concat = 13 nodes?
    // Wait, let's check.
    // t=0: Gather, MatMul(W), MatMul(R), Add, Add(B), Tanh
    // t=1: Gather, MatMul(W), MatMul(R), Add, Add(B), Tanh
    // Plus 1 initial constant for prevH if no initial state.
    // Plus 1 Concat.
    // Total = 1 (Const) + 6 + 6 + 1 (Concat) = 14 nodes.
    expect(nodes.length).toBe(14);
    expect(nodes[0].opType).toBe('Constant');
    expect(nodes[nodes.length - 1].opType).toBe('Concat');
  });

  test('emitRNNBase with unrolling and initial state', () => {
    const nodes = emitRNNBase('RNN', 'in', 'out', 'w', 'r', undefined, ['h0'], 'rnn_unroll_h0', {
      returnSequences: false,
      returnState: false,
      goBackwards: true,
      stateful: false,
      unroll: true,
      timeSteps: 2,
    });
    // t=0, t=1 (Gather, MatMul, MatMul, Add, Tanh) = 5 nodes each.
    // Identity at end.
    // Total = 10 + 1 = 11 nodes.
    expect(nodes.length).toBe(11);
    expect(nodes[10].opType).toBe('Identity');
  });

  test('emitBidirectional', () => {
    const nodesConcat = emitBidirectional(
      'LSTM',
      'in',
      'out',
      'w',
      'r',
      'b',
      'w2',
      'r2',
      'b2',
      ['h0', 'c0', 'hb', 'cb'],
      'bilstm1',
      {
        returnSequences: true,
        returnState: false,
        goBackwards: false,
        stateful: false,
        mergeMode: 'concat',
        linearBeforeReset: 1,
      },
    );
    expect(nodesConcat[0].opType).toBe('LSTM');
    expect(nodesConcat[0].attributes?.find((a) => a.name === 'direction')?.s).toBe('bidirectional');
    expect(nodesConcat[1].opType).toBe('Reshape');

    const nodesSum = emitBidirectional(
      'GRU',
      'in',
      'out',
      'w',
      'r',
      'b',
      'w2',
      'r2',
      'b2',
      [],
      'bigru',
      {
        returnSequences: false,
        returnState: false,
        goBackwards: false,
        stateful: false,
        mergeMode: 'sum',
      },
    );
    expect(nodesSum[1].opType).toBe('ReduceSum');
    expect(nodesSum[1].attributes?.find((a) => a.name === 'axes')?.ints).toEqual([0]);

    // test sum with returnSequences: true
    const nodesSumSeq = emitBidirectional(
      'GRU',
      'in',
      'out',
      'w',
      'r',
      'b',
      'w2',
      'r2',
      'b2',
      [],
      'bigru_seq',
      {
        returnSequences: true,
        returnState: false,
        goBackwards: false,
        stateful: false,
        mergeMode: 'sum',
      },
    );
    expect(nodesSumSeq[1].attributes?.find((a) => a.name === 'axes')?.ints).toEqual([2]);

    const nodesAve = emitBidirectional(
      'RNN',
      'in',
      'out',
      'w',
      'r',
      'b',
      'w2',
      'r2',
      'b2',
      [],
      'birnn1',
      {
        returnSequences: true,
        returnState: false,
        goBackwards: false,
        stateful: false,
        mergeMode: 'ave',
      },
    );
    expect(nodesAve[1].opType).toBe('ReduceMean');
    expect(nodesAve[1].attributes?.find((a) => a.name === 'axes')?.ints).toEqual([2]);

    // test ave with returnSequences: false
    const nodesAveState = emitBidirectional(
      'RNN',
      'in',
      'out',
      'w',
      'r',
      'b',
      'w2',
      'r2',
      'b2',
      [],
      'birnn1_state',
      {
        returnSequences: false,
        returnState: false,
        goBackwards: false,
        stateful: false,
        mergeMode: 'ave',
      },
    );
    expect(nodesAveState[1].attributes?.find((a) => a.name === 'axes')?.ints).toEqual([0]);

    const nodesMul = emitBidirectional(
      'RNN',
      'in',
      'out',
      'w',
      'r',
      undefined,
      'w2',
      'r2',
      undefined,
      [],
      'birnn2',
      {
        returnSequences: false,
        returnState: false,
        goBackwards: false,
        stateful: false,
        mergeMode: 'mul',
      },
    );
    expect(nodesMul[1].opType).toBe('Identity');

    // hit the `forwardBName || ''` branch
    const nodesBiInitial = emitBidirectional(
      'RNN',
      'in',
      'out',
      'w',
      'r',
      undefined,
      'w2',
      'r2',
      undefined,
      ['h0'],
      'birnn3',
      {
        returnSequences: false,
        returnState: false,
        goBackwards: false,
        stateful: false,
        mergeMode: 'concat',
      },
    );
    expect(nodesBiInitial[0].inputs).toEqual(['in', 'w', 'r', '', '', 'h0']);
  });

  test('reorderLSTMGates', () => {
    const weights = new Float32Array(8); // hiddenSize 2 (4 gates * 2 = 8)
    for (let i = 0; i < 8; i++) weights[i] = i;
    // Keras: i: 0, 1 | f: 2, 3 | c: 4, 5 | o: 6, 7
    // ONNX: i, o, f, c => 0, 1, 6, 7, 2, 3, 4, 5
    const reordered = reorderLSTMGates(weights, 2);
    expect(reordered).toEqual(new Float32Array([0, 1, 6, 7, 2, 3, 4, 5]));
  });

  test('reorderGRUGates', () => {
    const weights = new Float32Array([1, 2, 3]);
    const reordered = reorderGRUGates(weights, 1);
    expect(reordered).toEqual(new Float32Array([1, 2, 3]));
  });
});
