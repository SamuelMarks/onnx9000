import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Graph, Node, Tensor, Attribute } from '@onnx9000/core';
import { WebNNCompiler } from '../src/providers/webnn/compiler.js';
import { WebNNContextManager } from '../src/providers/webnn/context.js';

class MockMLGraphBuilder {
  input = vi.fn().mockReturnValue({});
  constant = vi.fn().mockReturnValue({});
  build = vi.fn().mockResolvedValue({});

  add = vi.fn().mockReturnValue({});
  sub = vi.fn().mockReturnValue({});
  mul = vi.fn().mockReturnValue({});
  div = vi.fn().mockReturnValue({});
  max = vi.fn().mockReturnValue({});
  min = vi.fn().mockReturnValue({});
  pow = vi.fn().mockReturnValue({});
  abs = vi.fn().mockReturnValue({});
  ceil = vi.fn().mockReturnValue({});
  floor = vi.fn().mockReturnValue({});
  exp = vi.fn().mockReturnValue({});
  log = vi.fn().mockReturnValue({});
  cos = vi.fn().mockReturnValue({});
  sin = vi.fn().mockReturnValue({});
  tan = vi.fn().mockReturnValue({});
  acos = vi.fn().mockReturnValue({});
  asin = vi.fn().mockReturnValue({});
  atan = vi.fn().mockReturnValue({});
  sqrt = vi.fn().mockReturnValue({});
  erf = vi.fn().mockReturnValue({});
  sign = vi.fn().mockReturnValue({});
  neg = vi.fn().mockReturnValue({});
  relu = vi.fn().mockReturnValue({});
  sigmoid = vi.fn().mockReturnValue({});
  tanh = vi.fn().mockReturnValue({});
  softmax = vi.fn().mockReturnValue({});
  leakyRelu = vi.fn().mockReturnValue({});
  elu = vi.fn().mockReturnValue({});
  hardSigmoid = vi.fn().mockReturnValue({});
  softplus = vi.fn().mockReturnValue({});
  softsign = vi.fn().mockReturnValue({});
  gelu = vi.fn().mockReturnValue({});
  prelu = vi.fn().mockReturnValue({});
  clamp = vi.fn().mockReturnValue({});
  matmul = vi.fn().mockReturnValue({});
  gemm = vi.fn().mockReturnValue({});
  reshape = vi.fn().mockReturnValue({});
  transpose = vi.fn().mockReturnValue({});
  concat = vi.fn().mockReturnValue({});
  split = vi.fn().mockReturnValue([{}, {}]);
  expand = vi.fn().mockReturnValue({});
  gather = vi.fn().mockReturnValue({});
  cast = vi.fn().mockReturnValue({});
  slice = vi.fn().mockReturnValue({});
  pad = vi.fn().mockReturnValue({});
  conv2d = vi.fn().mockReturnValue({});
  convTranspose2d = vi.fn().mockReturnValue({});
  maxPool2d = vi.fn().mockReturnValue({});
  averagePool2d = vi.fn().mockReturnValue({});
  reduceMean = vi.fn().mockReturnValue({});
  reduceSum = vi.fn().mockReturnValue({});
  reduceMax = vi.fn().mockReturnValue({});
  reduceMin = vi.fn().mockReturnValue({});
  reduceProduct = vi.fn().mockReturnValue({});
  reduceL1 = vi.fn().mockReturnValue({});
  reduceL2 = vi.fn().mockReturnValue({});
  reduceLogSumExp = vi.fn().mockReturnValue({});
  argMax = vi.fn().mockReturnValue({});
  argMin = vi.fn().mockReturnValue({});
  batchNormalization = vi.fn().mockReturnValue({});
  instanceNormalization = vi.fn().mockReturnValue({});
  layerNormalization = vi.fn().mockReturnValue({});
  equal = vi.fn().mockReturnValue({});
  greater = vi.fn().mockReturnValue({});
  greaterOrEqual = vi.fn().mockReturnValue({});
  lesser = vi.fn().mockReturnValue({});
  lesserOrEqual = vi.fn().mockReturnValue({});
  logicalNot = vi.fn().mockReturnValue({});
  logicalAnd = vi.fn().mockReturnValue({});
  logicalOr = vi.fn().mockReturnValue({});
  logicalXor = vi.fn().mockReturnValue({});
  where = vi.fn().mockReturnValue({});
  quantizeLinear = vi.fn().mockReturnValue({});
  dequantizeLinear = vi.fn().mockReturnValue({});
  scaledDotProductAttention = vi.fn().mockReturnValue({});
}

describe('WebNNCompiler mapping', () => {
  let builder: Object;
  let graph: Graph;

  beforeEach(() => {
    builder = new MockMLGraphBuilder();
    graph = new Graph('test');
    // Prepopulate generic inputs
    graph.inputs.push({ name: 'in1', shape: [1], id: 'i1', dtype: 'float32' });
    graph.inputs.push({ name: 'in2', shape: [1], id: 'i2', dtype: 'float32' });
    graph.outputs.push({ name: 'out1', shape: [1], id: 'o1', dtype: 'float32' } as Object);
  });

  const simpleOps = [
    ['Add', 'add'],
    ['Sub', 'sub'],
    ['Mul', 'mul'],
    ['Div', 'div'],
    ['Max', 'max'],
    ['Min', 'min'],
    ['Pow', 'pow'],
    ['Abs', 'abs'],
    ['Ceil', 'ceil'],
    ['Floor', 'floor'],
    ['Exp', 'exp'],
    ['Log', 'log'],
    ['Cos', 'cos'],
    ['Sin', 'sin'],
    ['Tan', 'tan'],
    ['Acos', 'acos'],
    ['Asin', 'asin'],
    ['Atan', 'atan'],
    ['Sqrt', 'sqrt'],
    ['Erf', 'erf'],
    ['Sign', 'sign'],
    ['Neg', 'neg'],
    ['Relu', 'relu'],
    ['Sigmoid', 'sigmoid'],
    ['Tanh', 'tanh'],
    ['Softmax', 'softmax'],
    ['LeakyRelu', 'leakyRelu'],
    ['Elu', 'elu'],
    ['HardSigmoid', 'hardSigmoid'],
    ['Softplus', 'softplus'],
    ['Softsign', 'softsign'],
    ['Gelu', 'gelu'],
    ['PRelu', 'prelu'],
    ['Clip', 'clamp'],
    ['MatMul', 'matmul'],
    ['Gemm', 'gemm'],
    ['Equal', 'equal'],
    ['Greater', 'greater'],
    ['GreaterOrEqual', 'greaterOrEqual'],
    ['Less', 'lesser'],
    ['LessOrEqual', 'lesserOrEqual'],
    ['Not', 'logicalNot'],
    ['And', 'logicalAnd'],
    ['Or', 'logicalOr'],
    ['Xor', 'logicalXor'],
    ['Where', 'where'],
    ['Conv', 'conv2d'],
    ['ConvTranspose', 'convTranspose2d'],
    ['MaxPool', 'maxPool2d'],
    ['AveragePool', 'averagePool2d'],
    ['GlobalMaxPool', 'maxPool2d'],
    ['GlobalAveragePool', 'averagePool2d'],
    ['ReduceMean', 'reduceMean'],
    ['ReduceSum', 'reduceSum'],
    ['ReduceMax', 'reduceMax'],
    ['ReduceMin', 'reduceMin'],
    ['ReduceProd', 'reduceProduct'],
    ['ReduceL1', 'reduceL1'],
    ['ReduceL2', 'reduceL2'],
    ['ReduceLogSumExp', 'reduceLogSumExp'],
    ['ArgMax', 'argMax'],
    ['ArgMin', 'argMin'],
    ['LayerNormalization', 'layerNormalization'],
    ['BatchNormalization', 'batchNormalization'],
    ['InstanceNormalization', 'instanceNormalization'],
    ['QuantizeLinear', 'quantizeLinear'],
    ['DequantizeLinear', 'dequantizeLinear'],
  ];

  it.each(simpleOps)('should map %s to %s', async (opType, method) => {
    if (['BatchNormalization', 'LayerNormalization', 'InstanceNormalization'].includes(opType)) {
      // Need specific inputs for norm
      graph.nodes.push(new Node(opType, ['in1', 'in1', 'in1', 'in1', 'in1'], ['out1']));
    } else if (opType === 'Where') {
      graph.inputs.push({ name: 'in3', shape: [1], id: 'i3', dtype: 'bool' });
      graph.nodes.push(new Node(opType, ['in1', 'in2', 'in3'], ['out1']));
    } else {
      const n = new Node(opType, ['in1', 'in2'], ['out1']);
      if (opType === 'Conv')
        n.attributes['auto_pad'] = new Attribute('auto_pad', 'STRING', 'SAME_UPPER');
      graph.nodes.push(n);
    }

    // Add default shape inputs for reshape/expand/etc
    const shapeData = new Int32Array([1]);
    graph.tensors['shape_tensor'] = new Tensor(
      'shape_tensor',
      [1],
      'int32',
      false,
      true,
      shapeData,
    );
    graph.tensors['in2'] = new Tensor('in2', [1], 'int32', false, true, shapeData);
    graph.tensors['in3'] = new Tensor('in3', [1], 'int32', false, true, shapeData);
    graph.tensors['in4'] = new Tensor('in4', [1], 'int32', false, true, shapeData);
    graph.tensors['in5'] = new Tensor('in5', [1], 'int32', false, true, shapeData);

    const compiler = new WebNNCompiler(graph, builder);
    await compiler.compile();
    expect(builder[method]).toHaveBeenCalled();
  });

  it('should map Attention to scaledDotProductAttention', async () => {
    graph.inputs.push({ name: 'in3', shape: [1], id: 'i3', dtype: 'float32' });
    graph.nodes.push(new Node('Attention', ['in1', 'in2', 'in3'], ['out1']));
    const compiler = new WebNNCompiler(graph, builder);
    await compiler.compile();
    expect(builder.scaledDotProductAttention).toHaveBeenCalled();
  });

  it('should map Slice correctly', async () => {
    const data = new Int32Array([0]);
    graph.tensors['starts'] = new Tensor('starts', [1], 'int32', false, true, data);
    graph.tensors['ends'] = new Tensor('ends', [1], 'int32', false, true, data);
    graph.initializers.push('starts', 'ends');
    graph.nodes.push(new Node('Slice', ['in1', 'starts', 'ends'], ['out1']));
    const compiler = new WebNNCompiler(graph, builder);
    await compiler.compile();
    expect(builder.slice).toHaveBeenCalled();
  });

  it('should map Split correctly', async () => {
    graph.outputs.push({ name: 'out2', shape: [1], id: 'o2', dtype: 'float32' } as Object);
    graph.nodes.push(new Node('Split', ['in1'], ['out1', 'out2']));
    const compiler = new WebNNCompiler(graph, builder);
    await compiler.compile();
    expect(builder.split).toHaveBeenCalled();
  });

  it('should handle reshape mapping', async () => {
    const data = new Int32Array([1, 1]);
    graph.tensors['shape'] = new Tensor('shape', [2], 'int32', false, true, data);
    graph.initializers.push('shape');
    graph.nodes.push(new Node('Reshape', ['in1', 'shape'], ['out1']));
    const compiler = new WebNNCompiler(graph, builder);
    await compiler.compile();
    expect(builder.reshape).toHaveBeenCalled();
  });

  it('should map Pad correctly', async () => {
    const data = new Int32Array([0, 0]);
    graph.tensors['pads'] = new Tensor('pads', [2], 'int32', false, true, data);
    graph.initializers.push('pads');
    graph.nodes.push(new Node('Pad', ['in1', 'pads'], ['out1']));
    const compiler = new WebNNCompiler(graph, builder);
    await compiler.compile();
    expect(builder.pad).toHaveBeenCalled();
  });

  it('should map Concat correctly', async () => {
    graph.nodes.push(new Node('Concat', ['in1', 'in2'], ['out1']));
    const compiler = new WebNNCompiler(graph, builder);
    await compiler.compile();
    expect(builder.concat).toHaveBeenCalled();
  });
});
