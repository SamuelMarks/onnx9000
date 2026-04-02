import { describe, it, expect, vi } from 'vitest';
import { Keras2OnnxConverter } from '../src/keras/index.js';
import { keras2onnx } from '../src/keras/api.js';
import { parseKerasH5 } from '../src/keras/h5-parser.js';
import { Keras2OnnxConverter } from '../src/keras/index.js';
import { parseKeras3Zip } from '../src/keras/keras3-parser.js';
import { readBrowserFile, fetchRemoteUrl } from '../src/keras/browser-io.js';
import { optimizeFusedOps, applyQuantization } from '../src/keras/optimizers.js';
import { downloadWeightShards } from '../src/keras/weight-loader.js';
import { emitConv, emitSeparableConv } from '../src/keras/emitters-conv.js';
import { emitPool, emitGlobalPool } from '../src/keras/emitters-pool.js';
import {
  emitBatchNormalization,
  emitLayerNormalization,
  emitReshape,
  emitFlatten,
  emitTranspose,
  emitPad,
} from '../src/keras/emitters-norm.js';
import {
  emitRNNBase,
  emitBidirectional,
  reorderLSTMGates,
  reorderGRUGates,
} from '../src/keras/emitters-rnn.js';
import { mapTfjsOpToOnnx } from '../src/keras/emitters-tfjs.js';
import { emitAttention, emitEmbedding } from '../src/keras/emitters-attention.js';
import { emitMerge, emitConcat, emitDot } from '../src/keras/emitters-merge.js';
import { zipSync } from 'fflate';

vi.mock('jsfive', () => ({
  File: class {
    constructor() {}
    get(name: string) {
      if (name === 'model_weights') {
        return {
          attrs: { weight_names: ['dense_1/kernel:0'] },
          keys: ['dense_1'],
          get: (key: string) => {
            if (key === 'dense_1') {
              return {
                attrs: { weight_names: ['kernel:0'] },
                keys: ['kernel:0'],
                get: (k: string) => ({ shape: [10], value: new Float32Array(10) }),
              };
            }
            throw new Error('Not found');
          },
        };
      }
      return {
        attrs: { model_config: JSON.stringify({ config: { layers: [] } }) },
        keys: [],
        get: () => {
          throw new Error('Not found');
        },
      };
    }
    attrs = {
      model_config: JSON.stringify({
        weightsManifest: [],
        modelTopology: {
          class_name: 'Sequential',
          config: { layers: [] },
        },
        weightsManifest: [],
      }),
      keras_version: '2.8.0',
      backend: 'tensorflow',
    };
  },
}));

describe('Keras Module', () => {
  const baseJsonStr = JSON.stringify({
    format: 'layers-model',
    weightsManifest: [],
    modelTopology: {
      class_name: 'Sequential',
      config: { layers: [{ class_name: 'Dense', config: { units: 10 } }] },
    },
    weightsManifest: [],
  });

  it('should test keras2onnx with string', async () => {
    const result = await keras2onnx(baseJsonStr);
    expect(result).toBeInstanceOf(Uint8Array);
  });

  it('should test keras2onnx with buffer', async () => {
    const buffer = new ArrayBuffer(10);
    const result = await keras2onnx(buffer);
    expect(result).toBeInstanceOf(Uint8Array);
  });

  it('should test h5-parser', () => {
    const buffer = new ArrayBuffer(10);
    const result = parseKerasH5(buffer);
    expect(result).toBeDefined();
    expect(result.modelConfig).toBeDefined();
  });

  it('should test keras3-parser', () => {
    const files = {
      'config.json': new TextEncoder().encode(JSON.stringify({ class_name: 'Sequential' })),
      'metadata.json': new TextEncoder().encode(JSON.stringify({ keras_version: '3.0' })),
      'model.weights.h5': new Uint8Array([1, 2, 3]),
      'weights/model.weights.safetensors': new Uint8Array([4, 5, 6]),
    };
    const zipped = zipSync(files);
    const result = parseKeras3Zip(zipped);
    expect(result.config.class_name).toBe('Sequential');
    expect(result.metadata.keras_version).toBe('3.0');
    expect(result.weightsH5).toBeDefined();
    expect(result.weightsSafetensors).toBeDefined();
  });

  it('should test keras3-parser error handling', () => {
    const zipped = zipSync({ 'other.txt': new Uint8Array([1]) });
    expect(() => parseKeras3Zip(zipped)).toThrow('Invalid Keras 3 format: missing config.json');
  });

  it('should test browser-io readBrowserFile', async () => {
    const blob = new Blob(['test']);
    const buffer = await readBrowserFile(blob as File);
    expect(buffer.byteLength).toBe(4);
  });

  it('should test browser-io fetchRemoteUrl', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      arrayBuffer: () => Promise.resolve(new ArrayBuffer(10)),
    });
    const result = await fetchRemoteUrl('http://test.com/weights.bin');
    expect(result.byteLength).toBe(10);

    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      statusText: 'Not Found',
    });
    await expect(fetchRemoteUrl('http://test.com/weights.bin')).rejects.toThrow(
      'Failed to fetch remote model: Not Found',
    );
  });

  it('should test optimizers optimizeFusedOps', () => {
    const nodes = [
      { opType: '_FusedConv2D', name: 'conv', inputs: ['in'], outputs: ['out'], attributes: [] },
      { opType: '_FusedMatMul', name: 'matmul', inputs: ['in'], outputs: ['out'], attributes: [] },
      { opType: 'StopGradient', name: 'stop', inputs: ['in'], outputs: ['out'], attributes: [] },
      { opType: 'Dense', name: 'dense', inputs: ['in'], outputs: ['out'], attributes: [] },
    ];
    const optimized = optimizeFusedOps(nodes);
    expect(optimized.length).toBe(4);
    expect(optimized[0].opType).toBe('Conv');
    expect(optimized[1].opType).toBe('Relu');
    expect(optimized[2].opType).toBe('MatMul');
    expect(optimized[3].opType).toBe('Dense');
  });

  it('should test optimizers applyQuantization', () => {
    const weights = [{ name: 'w', dtype: 'float32', data: new Float32Array(10) }];
    const quantized = applyQuantization(weights, 'fp16');
    expect(quantized[0].dtype).toBe('fp16');
  });

  it('should test weight-loader downloadWeightShards', async () => {
    const manifest = [
      {
        paths: ['shard1.bin'],
        weights: [
          { name: 'w', shape: [2, 2], dtype: 'float32' as any },
          { name: 'w2', shape: [2, 2], dtype: 'int32' as any },
          { name: 'w3', shape: [2, 2], dtype: 'uint8' as any },
          { name: 'w4', shape: [2, 2], dtype: 'bool' as any },
          { name: 'w5', shape: [2, 2], dtype: 'complex64' as any },
          { name: 'w6', shape: [2, 2], dtype: 'float16' as any },
        ],
      },
    ];
    // 4*4 + 4*4 + 4 + 4 + 4*8 + 4*2 = 16 + 16 + 4 + 4 + 32 + 8 = 80 bytes
    const fetcher = async () => new Uint8Array(80).buffer;
    const weights = await downloadWeightShards(manifest, 'http://test.com', fetcher);
    expect(weights.length).toBe(6);
  });

  it('should test weight-loader unsupported string dtype', async () => {
    const manifest = [
      {
        paths: ['shard1.bin'],
        weights: [{ name: 'w', shape: [2, 2], dtype: 'string' as any }],
      },
    ];
    const fetcher = async () => new Uint8Array(16).buffer;
    await expect(downloadWeightShards(manifest, 'http://test.com', fetcher)).rejects.toThrow();
  });

  it('should test weight-loader unknown dtype', async () => {
    const manifest = [
      {
        paths: ['shard1.bin'],
        weights: [{ name: 'w', shape: [2, 2], dtype: 'unknown' as any }],
      },
    ];
    const fetcher = async () => new Uint8Array(16).buffer;
    await expect(downloadWeightShards(manifest, 'http://test.com', fetcher)).rejects.toThrow();
  });

  it('should test weight-loader default fetcher success and failure', async () => {
    const manifest = [
      {
        paths: ['shard1.bin'],
        weights: [{ name: 'w', shape: [2, 2], dtype: 'float32' as any }],
      },
    ];

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      arrayBuffer: () => Promise.resolve(new Uint8Array(16).buffer),
    });

    const weights = await downloadWeightShards(manifest, 'http://test.com');
    expect(weights.length).toBe(1);

    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
    });
    await expect(downloadWeightShards(manifest, 'http://test.com')).rejects.toThrow();
  });

  it('should test Keras2OnnxConverter layer translation', () => {
    const modelJson = JSON.stringify({
      format: 'layers-model',
      weightsManifest: [],
      modelTopology: {
        class_name: 'Sequential',
        config: {
          layers: [
            { class_name: 'InputLayer', config: {} },
            {
              class_name: 'Dense',
              config: { units: 10, activation: 'relu', use_bias: false },
              inboundNodes: [['input_1']],
            },
            {
              class_name: 'Activation',
              config: { activation: 'sigmoid' },
              inboundNodes: [['dense_1']],
            },
            {
              class_name: 'Conv2D',
              config: {
                filters: 32,
                kernel_size: [3, 3],
                strides: [1, 1],
                padding: 'same',
                dilation_rate: [1, 1],
              },
              inboundNodes: [['act_1']],
            },
            {
              class_name: 'MaxPooling2D',
              config: { pool_size: [2, 2], strides: [2, 2], padding: 'valid' },
              inboundNodes: [['conv2d_1']],
            },
            {
              class_name: 'AveragePooling2D',
              config: { pool_size: [2, 2], strides: [2, 2], padding: 'valid' },
              inboundNodes: [['maxpool_1']],
            },
            {
              class_name: 'GlobalAveragePooling2D',
              config: { keepdims: true },
              inboundNodes: [['avgpool_1']],
            },
            {
              class_name: 'GlobalMaxPooling2D',
              config: { keepdims: false },
              inboundNodes: [['gavgpool_1']],
            },
            { class_name: 'Flatten', config: {}, inboundNodes: [['gmaxpool_1']] },
            { class_name: 'Dropout', config: { rate: 0.5 }, inboundNodes: [['flatten_1']] },
            { class_name: 'LeakyReLU', config: { alpha: 0.2 }, inboundNodes: [['dropout_1']] },
            { class_name: 'PReLU', config: {}, inboundNodes: [['leaky_relu_1']] },
            { class_name: 'ELU', config: { alpha: 1.0 }, inboundNodes: [['prelu_1']] },
            { class_name: 'ThresholdedReLU', config: { theta: 1.0 }, inboundNodes: [['elu_1']] },
            { class_name: 'Softmax', config: { axis: -1 }, inboundNodes: [['thresholded_relu_1']] },
            {
              class_name: 'QConv2D',
              config: {
                strides: [1, 1],
                padding: 'same',
                dilation_rate: [1, 1],
                kernel_size: [3, 3],
              },
              inboundNodes: [['softmax_1']],
            },
            {
              class_name: 'LayerNormalization',
              config: { axis: -1 },
              inboundNodes: [['qconv2d_1']],
            },
            {
              class_name: 'UnitNormalization',
              config: { axis: -1 },
              inboundNodes: [['layernorm_1']],
            },
            {
              class_name: 'GroupNormalization',
              config: { groups: 32 },
              inboundNodes: [['unitnorm_1']],
            },
            {
              class_name: 'Embedding',
              config: { input_dim: 100, output_dim: 32 },
              inboundNodes: [['groupnorm_1']],
            },
            { class_name: 'Permute', config: { dims: [2, 1] }, inboundNodes: [['embedding_1']] },
            {
              class_name: 'Reshape',
              config: { target_shape: [10, 10] },
              inboundNodes: [['permute_1']],
            },
            {
              class_name: 'Rescaling',
              config: { scale: 2.0, offset: 1.0 },
              inboundNodes: [['reshape_1']],
            },
            {
              class_name: 'Resizing',
              config: { height: 64, width: 64, interpolation: 'nearest' },
              inboundNodes: [['rescaling_1']],
            },
            {
              class_name: 'CenterCrop',
              config: { height: 32, width: 32 },
              inboundNodes: [['resizing_1']],
            },
            { class_name: 'Add', config: {}, inboundNodes: [['centercrop_1'], ['resizing_1']] },
            { class_name: 'Subtract', config: {}, inboundNodes: [['add_1'], ['centercrop_1']] },
            { class_name: 'Multiply', config: {}, inboundNodes: [['sub_1'], ['add_1']] },
            { class_name: 'Minimum', config: {}, inboundNodes: [['mul_1'], ['sub_1']] },
            { class_name: 'Maximum', config: {}, inboundNodes: [['min_1'], ['mul_1'], ['sub_1']] },
            {
              class_name: 'Concatenate',
              config: { axis: -1 },
              inboundNodes: [['max_1'], ['min_1']],
            },
            { class_name: 'Average', config: {}, inboundNodes: [['concat_1'], ['max_1']] },
            { class_name: 'Dot', config: { axes: 1 }, inboundNodes: [['avg_1'], ['concat_1']] },
            {
              class_name: 'EinsumDense',
              config: { equation: 'ab,bc->ac', bias_axes: 'c' },
              inboundNodes: [['dot_1']],
            },
            { class_name: 'GaussianNoise', config: { stddev: 0.1 }, inboundNodes: [['einsum_1']] },
            {
              class_name: 'RandomFlip',
              config: { mode: 'horizontal' },
              inboundNodes: [['noise_1']],
            },
            { class_name: 'UnknownLayer', config: {}, inboundNodes: [['flip_1']] },
          ],
        },
      },
      weightsManifest: [],
    });
    const converter = new Keras2OnnxConverter(modelJson);
    const result = converter.convert();
    expect(result).toBeInstanceOf(Uint8Array);
  });

  it('should test emitters-conv', () => {
    emitConv('Conv', 'in', 'out', 'w', 'b', 'name', {
      padding: 'same',
      kernelShape: [3, 3],
      strides: [1, 1],
      dilations: [1, 1],
      activation: 'relu',
    });
    emitConv('Conv', 'in', 'out', 'w', 'b', 'name', {
      padding: 'valid',
      kernelShape: [3, 3],
      strides: [1, 1],
      dilations: [1, 1],
      activation: 'linear',
    });
    emitSeparableConv('in', 'out', 'depth_w', 'point_w', 'b', 'name', {
      padding: 'same',
      kernelShape: [3, 3],
      strides: [1, 1],
      dilations: [1, 1],
      activation: 'relu',
    });
  });

  it('should test emitters-pool', () => {
    emitPool('Max', 'in', 'out', 'name', { padding: 'same', poolSize: [2, 2], strides: [2, 2] });
    emitGlobalPool('Average', 'in', 'out', 'name', { keepDims: true });
  });

  it('should test emitters-norm', () => {
    emitBatchNormalization('in', 'out', 'gamma', 'beta', 'mean', 'var', 'name', {
      epsilon: 1e-5,
      momentum: 0.9,
    });
    emitLayerNormalization('in', 'out', 'gamma', 'beta', 'name', { epsilon: 1e-5, axis: -1 });
    emitReshape('in', 'out', 'name', [1, 2, 3]);
    emitFlatten('in', 'out', 'name', 1);
    emitTranspose('in', 'out', 'name', [0, 2, 1]);
    emitPad(
      'in',
      'out',
      'name',
      [
        [1, 1],
        [1, 1],
      ],
      'constant',
      0,
    );
  });

  it('should test emitters-rnn', () => {
    emitRNNBase('LSTM', 'in', 'out', 'w', 'r', 'b', ['h_0', 'c_0'], 'name', {
      returnSequences: true,
      returnState: false,
      goBackwards: false,
      stateful: false,
      linearBeforeReset: 1,
    });
    emitRNNBase('GRU', 'in', 'out', 'w', 'r', undefined, [], 'name', {
      returnSequences: false,
      returnState: true,
      goBackwards: true,
      stateful: true,
    });

    emitBidirectional('LSTM', 'in', 'out', 'fw', 'fr', 'fb', 'bw', 'br', 'bb', ['h_0'], 'name', {
      mergeMode: 'concat',
      returnSequences: true,
      returnState: false,
      goBackwards: false,
      stateful: false,
      linearBeforeReset: 1,
    });
    emitBidirectional(
      'LSTM',
      'in',
      'out',
      'fw',
      'fr',
      undefined,
      'bw',
      'br',
      undefined,
      [],
      'name',
      {
        mergeMode: 'sum',
        returnSequences: false,
        returnState: false,
        goBackwards: false,
        stateful: false,
      },
    );
    emitBidirectional('LSTM', 'in', 'out', 'fw', 'fr', 'fb', 'bw', 'br', 'bb', [], 'name', {
      mergeMode: 'ave',
      returnSequences: false,
      returnState: false,
      goBackwards: false,
      stateful: false,
    });
    emitBidirectional('LSTM', 'in', 'out', 'fw', 'fr', 'fb', 'bw', 'br', 'bb', [], 'name', {
      mergeMode: 'mul',
      returnSequences: false,
      returnState: false,
      goBackwards: false,
      stateful: false,
    });

    reorderLSTMGates(new Float32Array(16), 4);
    reorderGRUGates(new Float32Array(12), 4);
  });

  it('should test emitters-tfjs', () => {
    mapTfjsOpToOnnx('Add', ['a', 'b'], 'out', 'name');
    mapTfjsOpToOnnx('Sub', ['a', 'b'], 'out', 'name');
    mapTfjsOpToOnnx('Mul', ['a', 'b'], 'out', 'name');
    mapTfjsOpToOnnx('Div', ['a', 'b'], 'out', 'name');
    mapTfjsOpToOnnx('MatMul', ['a', 'b'], 'out', 'name');
    mapTfjsOpToOnnx('Square', ['in'], 'out', 'name');
    mapTfjsOpToOnnx('Sqrt', ['in'], 'out', 'name');
    mapTfjsOpToOnnx('Exp', ['in'], 'out', 'name');
    mapTfjsOpToOnnx('Log', ['in'], 'out', 'name');
    mapTfjsOpToOnnx('Maximum', ['a', 'b'], 'out', 'name');
    mapTfjsOpToOnnx('Minimum', ['a', 'b'], 'out', 'name');
    mapTfjsOpToOnnx('Sum', ['in'], 'out', 'name');
    mapTfjsOpToOnnx('Mean', ['in'], 'out', 'name');
    mapTfjsOpToOnnx('Max', ['in'], 'out', 'name');
    mapTfjsOpToOnnx('Min', ['in'], 'out', 'name');
    mapTfjsOpToOnnx('ArgMax', ['in'], 'out', 'name');
    mapTfjsOpToOnnx('ArgMin', ['in'], 'out', 'name');
    mapTfjsOpToOnnx('Split', ['in'], 'out', 'name');
    mapTfjsOpToOnnx('Concat', ['a', 'b'], 'out', 'name');
    mapTfjsOpToOnnx('Slice', ['in'], 'out', 'name');
    mapTfjsOpToOnnx('Gather', ['in'], 'out', 'name');
    mapTfjsOpToOnnx('GatherNd', ['in'], 'out', 'name');
    mapTfjsOpToOnnx('Where', ['in'], 'out', 'name');
    mapTfjsOpToOnnx('TensorScatterUpdate', ['in'], 'out', 'name');
    mapTfjsOpToOnnx('ResizeBilinear', ['in'], 'out', 'name');
    mapTfjsOpToOnnx('ResizeNearestNeighbor', ['in'], 'out', 'name');

    expect(() => mapTfjsOpToOnnx('UnsupportedOp', ['in'], 'out', 'name')).toThrow(
      'Unsupported TF.js Op: UnsupportedOp',
    );
  });

  it('should test emitters-attention', () => {
    emitAttention('q', 'v', 'k', 'out', 'name', { useScale: true, causal: false });
    emitAttention('q', 'v', 'k', 'out', 'name', { useScale: false, causal: true });
    emitEmbedding('in', 'out', 'w', 'name');
  });

  it('should test emitters-merge', () => {
    emitMerge('Add', ['a', 'b'], 'out', 'name');
    emitMerge('Add', ['a', 'b', 'c'], 'out', 'name');
    expect(() => emitMerge('Add', ['a'], 'out', 'name')).toThrow(
      'Merge layer requires at least 2 inputs, got 1',
    );

    emitMerge('Sub', ['a', 'b'], 'out', 'name');
    emitMerge('Mul', ['a', 'b'], 'out', 'name');
    emitMerge('Mean', ['a', 'b'], 'out', 'name');
    emitMerge('Max', ['a', 'b'], 'out', 'name');
    emitMerge('Min', ['a', 'b'], 'out', 'name');

    emitConcat(['a', 'b'], 'out', 'name', 1);
    emitDot('a', 'b', 'out', [1, 1], 'name');
  });
});

describe('Keras2OnnxConverter additional layers', () => {
  it('should cover GlobalPooling, Flatten, and Dropout', () => {
    const mockJson = JSON.stringify({
      format: 'layers-model',
      generatedBy: 'keras v2.0.0',
      convertedBy: 'TensorFlow.js',
      weightsManifest: [],
      modelTopology: {
        class_name: 'Sequential',
        config: {
          layers: [
            { class_name: 'GlobalAveragePooling2D', config: { name: 'g_pool', keepdims: true } },
            { class_name: 'Flatten', config: { name: 'flat' } },
            { class_name: 'Dropout', config: { name: 'drop' } },
          ],
        },
      },
    });

    const conv = new Keras2OnnxConverter(mockJson);
    const bytes = conv.convert();
    expect(bytes).toBeDefined();

    // We can also trigger the translateLayer directly if needed, but convert() loops through topology
  });
});
