import { test, expect } from 'vitest';
import { parseMxNetSymbol, parseMxNetParams } from '../../src/mmdnn/mxnet/parser.js';
import { MxNetMapper } from '../../src/mmdnn/mxnet/mapper.js';
import { Graph, Node } from '@onnx9000/core';

test('MxNetMapper - Unknown and null', () => {
  const mapper = new MxNetMapper();
  const graph = new Graph();
  expect(mapper.map({ op: 'null', name: 'data' }, graph)).toEqual([]);
  expect(mapper.map({ op: 'UnknownOp', name: 'unk' }, graph)).toEqual([]);
});

test('MxNetMapper - Convolution variations', () => {
  const mapper = new MxNetMapper();
  const graph = new Graph();
  let nodes = mapper.map({ op: 'Convolution', name: 'conv2' }, graph);
  expect(nodes[0].opType).toBe('Conv');
  expect(nodes[0].attributes['kernel_shape']).toBeUndefined();
  nodes = mapper.map(
    { op: 'Convolution', name: 'conv3', attrs: { pad: '(1, 1, 1, 1)', dilate: '(2, 2)' } },
    graph,
  );
  expect(nodes[0].attributes['pads'].value).toEqual([1, 1, 1, 1]);
  expect(nodes[0].attributes['dilations'].value).toEqual([2, 2]);
  nodes = mapper.map(
    { op: 'Convolution', name: 'conv4', attrs: { pad: '1, 1', kernel: '(1, a)' } },
    graph,
  );
  expect(nodes[0].attributes['pads']).toBeUndefined();
  expect(nodes[0].attributes['kernel_shape'].value).toEqual([1]);
});

test('MxNetMapper - Activation variations', () => {
  const mapper = new MxNetMapper();
  const graph = new Graph();
  expect(
    mapper.map({ op: 'Activation', name: 'act', attrs: { act_type: 'sigmoid' } }, graph)[0].opType,
  ).toBe('Sigmoid');
  expect(
    mapper.map({ op: 'Activation', name: 'act', attrs: { act_type: 'tanh' } }, graph)[0].opType,
  ).toBe('Tanh');
  expect(
    mapper.map({ op: 'Activation', name: 'act', attrs: { act_type: 'unknown' } }, graph)[0].opType,
  ).toBe('Relu');
  expect(mapper.map({ op: 'Activation', name: 'act' }, graph)[0].opType).toBe('Relu');
});

test('MxNetMapper - Pooling variations', () => {
  const mapper = new MxNetMapper();
  const graph = new Graph();
  expect(
    mapper.map(
      {
        op: 'Pooling',
        name: 'pool',
        attrs: { pool_type: 'avg', pad: '(1, 1, 1, 1)', stride: '(2, 2)' },
      },
      graph,
    )[0].opType,
  ).toBe('AveragePool');
  expect(mapper.map({ op: 'Pooling', name: 'pool' }, graph)[0].opType).toBe('MaxPool');
  expect(
    mapper.map({ op: 'Pooling', name: 'pool', attrs: { pool_type: 'unknown' } }, graph)[0].opType,
  ).toBe('MaxPool');
});

test('MxNetMapper - BatchNorm variations', () => {
  const mapper = new MxNetMapper();
  const graph = new Graph();
  expect(mapper.map({ op: 'BatchNorm', name: 'bn' }, graph)[0].opType).toBe('BatchNormalization');
  const bnAttrs = mapper.map(
    { op: 'BatchNorm', name: 'bn', attrs: { eps: '1e-5', momentum: '0.9' } },
    graph,
  );
  expect(bnAttrs[0].attributes['epsilon'].value).toBeCloseTo(1e-5);
});

test('MxNetMapper - Reshape variations', () => {
  const mapper = new MxNetMapper();
  const graph = new Graph();
  expect(mapper.map({ op: 'Reshape', name: 'resh' }, graph)[0].opType).toBe('Reshape');
  expect(
    mapper.map({ op: 'Reshape', name: 'resh', attrs: { shape: '(1, 2, 3)' } }, graph)[0].opType,
  ).toBe('Reshape');
});

test('MxNetMapper - Concat variations', () => {
  const mapper = new MxNetMapper();
  const graph = new Graph();
  expect(mapper.map({ op: 'Concat', name: 'concat' }, graph)[0].opType).toBe('Concat');
});

test('MxNetMapper - LeakyReLU variations', () => {
  const mapper = new MxNetMapper();
  const graph = new Graph();
  expect(
    mapper.map({ op: 'LeakyReLU', name: 'lr', attrs: { act_type: 'leaky' } }, graph)[0].opType,
  ).toBe('LeakyRelu');
  expect(
    mapper.map({ op: 'LeakyReLU', name: 'lr', attrs: { act_type: 'elu' } }, graph)[0].opType,
  ).toBe('Elu');
  expect(
    mapper.map({ op: 'LeakyReLU', name: 'lr', attrs: { act_type: 'prelu' } }, graph)[0].opType,
  ).toBe('PRelu');
  expect(
    mapper.map({ op: 'LeakyReLU', name: 'lr', attrs: { act_type: 'unknown' } }, graph)[0].opType,
  ).toBe('LeakyRelu');
  expect(mapper.map({ op: 'LeakyReLU', name: 'lr' }, graph)[0].opType).toBe('LeakyRelu');
});

test('MxNetMapper - UpSampling variations', () => {
  const mapper = new MxNetMapper();
  const graph = new Graph();
  expect(
    mapper.map({ op: 'UpSampling', name: 'us', attrs: { sample_type: 'nearest' } }, graph)[0]
      .attributes['mode'].value,
  ).toBe('nearest');
  expect(mapper.map({ op: 'UpSampling', name: 'us' }, graph)[0].attributes['mode'].value).toBe(
    'linear',
  );
});

test('MxNetMapper - SliceChannel variations', () => {
  const mapper = new MxNetMapper();
  const graph = new Graph();
  expect(
    mapper.map({ op: 'SliceChannel', name: 'sc', attrs: { axis: '2' } }, graph)[0].opType,
  ).toBe('Split');
  expect(mapper.map({ op: 'SliceChannel', name: 'sc' }, graph)[0].opType).toBe('Split');
});

test('MxNetMapper - Crop', () => {
  const mapper = new MxNetMapper();
  const graph = new Graph();
  expect(mapper.map({ op: 'Crop', name: 'cr' }, graph)[0].opType).toBe('Slice');
});

test('MxNetMapper - Deconvolution', () => {
  const mapper = new MxNetMapper();
  const graph = new Graph();
  expect(mapper.map({ op: 'Deconvolution', name: 'dc' }, graph)[0].opType).toBe('ConvTranspose');
});

test('MxNetMapper - FullyConnected', () => {
  const mapper = new MxNetMapper();
  const graph = new Graph();
  expect(mapper.map({ op: 'FullyConnected', name: 'fc1' }, graph)[0].opType).toBe('Gemm');
});

test('MxNetMapper - Dropout', () => {
  const mapper = new MxNetMapper();
  const graph = new Graph();
  expect(mapper.map({ op: 'Dropout', name: 'drop' }, graph)[0].opType).toBe('Identity');
});

test('MxNetMapper - Misc Ops', () => {
  const mapper = new MxNetMapper();
  const graph = new Graph();
  expect(mapper.map({ op: 'elemwise_add', name: 'eadd' }, graph)[0].opType).toBe('Add');
  expect(mapper.map({ op: 'elemwise_sub', name: 'esub' }, graph)[0].opType).toBe('Sub');
  expect(mapper.map({ op: 'elemwise_mul', name: 'emul' }, graph)[0].opType).toBe('Mul');
  expect(mapper.map({ op: 'broadcast_add', name: 'badd' }, graph)[0].opType).toBe('Add');
  expect(mapper.map({ op: 'broadcast_mul', name: 'bmul' }, graph)[0].opType).toBe('Mul');
  expect(mapper.map({ op: 'SoftmaxOutput', name: 'sm' }, graph)[0].opType).toBe('Softmax');
  expect(mapper.map({ op: 'Flatten', name: 'flat' }, graph)[0].opType).toBe('Flatten');
});

test('MxNetMapper - Coverage for node.name and node.attrs fallbacks', () => {
  const mapper = new MxNetMapper();
  const graph = new Graph();
  const ops = [
    'Convolution',
    'FullyConnected',
    'Activation',
    'Pooling',
    'BatchNorm',
    'Dropout',
    'Flatten',
    'Reshape',
    'Concat',
    'elemwise_add',
    'elemwise_sub',
    'elemwise_mul',
    'broadcast_add',
    'broadcast_mul',
    'SoftmaxOutput',
    'LeakyReLU',
    'UpSampling',
    'SliceChannel',
    'Crop',
    'Deconvolution',
  ];
  for (const op of ops) {
    const nodes = mapper.map({ op }, graph);
    expect(nodes.length).toBeGreaterThan(0);
    expect(nodes[0].name).toBe('');
    expect(nodes[0].outputs[0]).toBe('');
  }
});

// Parser Tests
test('parseMxNetSymbol', () => {
  const jsonStr = JSON.stringify({
    nodes: [
      { op: 'null', name: 'data', inputs: [] },
      {
        op: 'Convolution',
        name: 'conv1',
        attrs: { kernel: '(3, 3)', stride: '(1, 1)', num_group: '1' },
        inputs: [[0, 0, 0]],
      },
    ],
    arg_nodes: [0],
    heads: [[1, 0, 0]],
  });
  const symbol = parseMxNetSymbol(jsonStr);
  expect(symbol.nodes[1].op).toBe('Convolution');
});

function buildMxNetParamBuffer(options: Object) {
  const { magic, includeZeroCount, count, arrays, numNames, names } = options;
  let size = 8;
  if (includeZeroCount) size += 16;
  else size += 8;
  for (let arr of arrays) {
    size += 4;
    if (arr.tensorMagic === 0xf993fac9) size += 4;
    size += 8; // ndim + dtypeCode
    size += arr.shape.length * (arr.tensorMagic === 0xf993fac9 ? 8 : 4);
    let elements = arr.shape.reduce((a: number, b: number) => a * b, 1);
    let bytesPerElement = arr.dtypeCode === 1 ? 8 : 4;
    size += elements * bytesPerElement;
  }
  size += 8;
  for (let name of names) size += 8 + new TextEncoder().encode(name).length;
  const buffer = new ArrayBuffer(size);
  const view = new DataView(buffer);
  let offset = 0;
  view.setBigUint64(offset, BigInt(magic), true);
  offset += 8;
  if (includeZeroCount) {
    view.setBigUint64(offset, 0n, true);
    offset += 8;
  }
  view.setBigUint64(offset, BigInt(count), true);
  offset += 8;
  for (let arr of arrays) {
    view.setUint32(offset, arr.tensorMagic, true);
    offset += 4;
    if (arr.tensorMagic === 0xf993fac9) {
      view.setUint32(offset, 0, true);
      offset += 4;
    }
    view.setUint32(offset, arr.shape.length, true);
    offset += 4;
    view.setInt32(offset, arr.dtypeCode, true);
    offset += 4;
    for (let d of arr.shape) {
      if (arr.tensorMagic === 0xf993fac9) {
        view.setBigInt64(offset, BigInt(d), true);
        offset += 8;
      } else {
        view.setUint32(offset, d, true);
        offset += 4;
      }
    }
    if (arr.dtypeCode === 0)
      for (let v of arr.data) {
        view.setFloat32(offset, v, true);
        offset += 4;
      }
    else if (arr.dtypeCode === 1)
      for (let v of arr.data) {
        view.setFloat64(offset, v, true);
        offset += 8;
      }
    else if (arr.dtypeCode === 4)
      for (let v of arr.data) {
        view.setInt32(offset, v, true);
        offset += 4;
      }
  }
  view.setBigUint64(offset, BigInt(numNames), true);
  offset += 8;
  for (let name of names) {
    const nameBytes = new TextEncoder().encode(name);
    view.setBigUint64(offset, BigInt(nameBytes.length), true);
    offset += 8;
    for (let i = 0; i < nameBytes.length; i++) view.setUint8(offset++, nameBytes[i]);
  }
  return new Uint8Array(buffer);
}

test('parseMxNetParams - v2 magic with zero reserved count and float32', () => {
  const buf = buildMxNetParamBuffer({
    magic: 0x0112,
    includeZeroCount: true,
    count: 1,
    arrays: [{ tensorMagic: 0xf993fac9, shape: [2, 2], dtypeCode: 0, data: [1.0, 2.0, 3.0, 4.0] }],
    numNames: 1,
    names: ['test1'],
  });
  expect(parseMxNetParams(buf)['test1'].data[3]).toBe(4.0);
});

test('parseMxNetParams - v1 magic, float64 and int32, direct count', () => {
  const buf = buildMxNetParamBuffer({
    magic: 0x0112,
    includeZeroCount: false,
    count: 2,
    arrays: [
      { tensorMagic: 0x0112, shape: [2], dtypeCode: 1, data: [1.5, 2.5] },
      { tensorMagic: 0x0112, shape: [3], dtypeCode: 4, data: [10, 20, 30] },
    ],
    numNames: 2,
    names: ['test_f64', 'test_i32'],
  });
  expect(parseMxNetParams(buf)['test_i32'].data[2]).toBe(30);
});

test('parseMxNetParams - Error Invalid params magic', () => {
  const buf = buildMxNetParamBuffer({
    magic: 0x1234,
    includeZeroCount: false,
    count: 0,
    arrays: [],
    numNames: 0,
    names: [],
  });
  expect(() => parseMxNetParams(buf)).toThrowError(/Invalid MXNet params magic/);
});

test('parseMxNetParams - Error Invalid NDArray block magic', () => {
  const buf = buildMxNetParamBuffer({
    magic: 0x0112,
    includeZeroCount: false,
    count: 1,
    arrays: [{ tensorMagic: 0x99999999, shape: [1], dtypeCode: 0, data: [1.0] }],
    numNames: 1,
    names: ['test1'],
  });
  expect(() => parseMxNetParams(buf)).toThrowError(/Invalid MXNet NDArray block magic/);
});

test('parseMxNetParams - Error Unsupported dtype', () => {
  const buf = buildMxNetParamBuffer({
    magic: 0x0112,
    includeZeroCount: false,
    count: 1,
    arrays: [{ tensorMagic: 0x0112, shape: [1], dtypeCode: 3, data: [1.0] }],
    numNames: 1,
    names: ['test1'],
  });
  expect(() => parseMxNetParams(buf)).toThrowError(/Unsupported MXNet dtype/);
});

test('parseMxNetParams - Error Names count mismatch', () => {
  const buf = buildMxNetParamBuffer({
    magic: 0x0112,
    includeZeroCount: false,
    count: 1,
    arrays: [{ tensorMagic: 0x0112, shape: [1], dtypeCode: 0, data: [1.0] }],
    numNames: 2,
    names: ['test1', 'test2'],
  });
  expect(() => parseMxNetParams(buf)).toThrowError(/Names count 2 does not match arrays count 1/);
});
