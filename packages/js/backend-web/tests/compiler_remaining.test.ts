import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Graph, Node, Tensor, Attribute } from '@onnx9000/core';
import { WebNNCompiler } from '../src/providers/webnn/compiler.js';

describe('WebNNCompiler remaining coverage', () => {
  let builder: any;

  beforeEach(() => {
    // @ts-ignore
    global.MLGraphBuilder = class {
      input() {
        return {};
      }
      constant() {
        return {};
      }
      build() {
        return Promise.resolve({});
      }
      add() {
        return {};
      }
      sub() {
        return {};
      }
      mul() {
        return {};
      }
      div() {
        return {};
      }
      pow() {
        return {};
      }
      reduceMean() {
        return {};
      }
      sqrt() {
        return {};
      }
      layerNormalization() {
        return {};
      }
      quantizeLinear() {
        return {};
      }
      dequantizeLinear() {
        return {};
      }
      scaledDotProductAttention() {
        return {};
      }
      matmul() {
        return {};
      }
      gemm() {
        return {};
      }
      softmax() {
        return {};
      }
      conv2d() {
        return {};
      }
      convTranspose2d() {
        return {};
      }
      maxPool2d() {
        return {};
      }
      averagePool2d() {
        return {};
      }
      pad() {
        return {};
      }
      slice() {
        return {};
      }
      split() {
        return [{}, {}];
      }
      concat() {
        return {};
      }
      gather() {
        return {};
      }
      cast() {
        return {};
      }
      reshape() {
        return {};
      }
      expand() {
        return {};
      }
      batchNormalization() {
        return {};
      }
      instanceNormalization() {
        return {};
      }
      argMax() {
        return {};
      }
      argMin() {
        return {};
      }
      reduceSum() {
        return {};
      }
    };
    builder = new (globalThis as any).MLGraphBuilder();
  });

  const compileNode = async (node: Node, setupGraph?: (g: Graph) => void) => {
    const g = new Graph('test');
    g.inputs.push({ name: 'in1', shape: [1], id: 'i1', dtype: 'float32' });
    g.inputs.push({ name: 'in2', shape: [1], id: 'i2', dtype: 'float32' });
    g.inputs.push({ name: 'in3', shape: [1], id: 'i3', dtype: 'float32' });
    g.inputs.push({ name: 'in4', shape: [1], id: 'i4', dtype: 'float32' });
    g.inputs.push({ name: 'in5', shape: [1], id: 'i5', dtype: 'float32' });
    if (setupGraph) setupGraph(g);
    g.nodes.push(node);
    g.outputs.push({
      name: node.outputs[0] || 'out',
      shape: [1],
      id: 'o1',
      dtype: 'float32',
    } as any);
    const compiler = new WebNNCompiler(g, builder);
    await compiler.compile();
  };

  it('Conv with autoPad SAME_UPPER and explicit pads', async () => {
    const n = new Node('Conv', ['in1', 'in2'], ['out']);
    n.attributes['auto_pad'] = new Attribute('auto_pad', 'STRING', 'SAME_UPPER');
    n.attributes['strides'] = new Attribute('strides', 'INTS', [1, 1]);
    n.attributes['dilations'] = new Attribute('dilations', 'INTS', [1, 1]);
    n.attributes['group'] = new Attribute('group', 'INT', 1);
    await compileNode(n);

    const n2 = new Node('Conv', ['in1', 'in2', 'in3'], ['out']);
    n2.attributes['auto_pad'] = new Attribute('auto_pad', 'STRING', 'SAME_LOWER');
    await compileNode(n2);

    const n3 = new Node('Conv', ['in1', 'in2'], ['out']);
    n3.attributes['auto_pad'] = new Attribute('auto_pad', 'STRING', 'VALID');
    await compileNode(n3);

    const n4 = new Node('Conv', ['in1', 'in2'], ['out']);
    n4.attributes['pads'] = new Attribute('pads', 'INTS', [1, 1, 1, 1]);
    await compileNode(n4);
  });

  it('ConvTranspose with autoPad SAME_UPPER and explicit pads', async () => {
    const n = new Node('ConvTranspose', ['in1', 'in2', 'in3'], ['out']);
    n.attributes['strides'] = new Attribute('strides', 'INTS', [1, 1]);
    n.attributes['dilations'] = new Attribute('dilations', 'INTS', [1, 1]);
    n.attributes['group'] = new Attribute('group', 'INT', 1);
    n.attributes['pads'] = new Attribute('pads', 'INTS', [1, 1, 1, 1]);
    n.attributes['output_padding'] = new Attribute('output_padding', 'INTS', [1, 1]);
    await compileNode(n);
  });

  it('Pool autoPads', async () => {
    const n = new Node('MaxPool', ['in1'], ['out']);
    n.attributes['kernel_shape'] = new Attribute('kernel_shape', 'INTS', [2, 2]);
    n.attributes['strides'] = new Attribute('strides', 'INTS', [1, 1]);
    n.attributes['dilations'] = new Attribute('dilations', 'INTS', [1, 1]);
    n.attributes['pads'] = new Attribute('pads', 'INTS', [1, 1, 1, 1]);
    n.attributes['ceil_mode'] = new Attribute('ceil_mode', 'INT', 1);
    n.attributes['auto_pad'] = new Attribute('auto_pad', 'STRING', 'SAME_UPPER');
    await compileNode(n);

    const n2 = new Node('AveragePool', ['in1'], ['out']);
    n2.attributes['auto_pad'] = new Attribute('auto_pad', 'STRING', 'SAME_LOWER');
    await compileNode(n2);
  });

  it('ArgMax/ArgMin keepdims', async () => {
    const n = new Node('ArgMax', ['in1'], ['out']);
    n.attributes['axis'] = new Attribute('axis', 'INT', 1);
    n.attributes['keepdims'] = new Attribute('keepdims', 'INT', 0);
    await compileNode(n);

    const n2 = new Node('ArgMin', ['in1'], ['out']);
    n2.attributes['axis'] = new Attribute('axis', 'INT', 1);
    await compileNode(n2);
  });

  it('Reduction fallback and axes extraction', async () => {
    const n = new Node('ReduceSum', ['in1', 'in2'], ['out']);
    await compileNode(n, (g) => {
      g.tensors['in2'] = new Tensor('in2', [1], 'int32', false, true, new Int32Array([0]));
    });
  });

  it('Normalization default fallbacks', async () => {
    const n = new Node('BatchNormalization', ['in1', 'in2', 'in3', 'in4', 'in5'], ['out']);
    n.attributes['epsilon'] = new Attribute('epsilon', 'FLOAT', 1e-4);
    await compileNode(n);

    const n2 = new Node('InstanceNormalization', ['in1', 'in2', 'in3'], ['out']);
    await compileNode(n2);

    const n3 = new Node('LayerNormalization', ['in1', 'in2', 'in3'], ['out']);
    n3.attributes['axis'] = new Attribute('axis', 'INT', 2);
    await compileNode(n3);
  });

  it('Gemm transB', async () => {
    const n = new Node('Gemm', ['in1', 'in2'], ['out']);
    n.attributes['alpha'] = new Attribute('alpha', 'FLOAT', 2.0);
    n.attributes['beta'] = new Attribute('beta', 'FLOAT', 2.0);
    n.attributes['transA'] = new Attribute('transA', 'INT', 1);
    n.attributes['transB'] = new Attribute('transB', 'INT', 1);
    await compileNode(n);
  });

  it('Quantization emulation fallbacks', async () => {
    const n = new Node('QuantizeLinear', ['in1', 'in2'], ['out']);
    await compileNode(n);

    const n2 = new Node('DequantizeLinear', ['in1', 'in2'], ['out']);
    await compileNode(n2);

    const n3 = new Node('DynamicQuantizeLinear', ['in1'], ['out']);
    await expect(compileNode(n3)).rejects.toThrow(
      'DynamicQuantizeLinear requires fallback emulation via min/max',
    );

    // Remove builder support to test emulation
    builder.dequantizeLinear = undefined;
    const n4 = new Node('DequantizeLinear', ['in1', 'in2'], ['out']);
    await compileNode(n4);
  });

  it('Attention manual decomp', async () => {
    builder.scaledDotProductAttention = undefined;
    const n = new Node('Attention', ['in1', 'in2', 'in3'], ['out']);
    await compileNode(n);
  });

  it('LayerNorm manual decomp', async () => {
    builder.layerNormalization = () => {
      throw new Error('Not supported');
    };
    const n = new Node('LayerNormalization', ['in1', 'in2', 'in3'], ['out']);
    await compileNode(n);
  });

  it('Pad explicit arrays', async () => {
    const n = new Node('Pad', ['in1', 'in2', 'in3'], ['out']);
    n.attributes['mode'] = new Attribute('mode', 'STRING', 'reflect');
    await compileNode(n, (g) => {
      g.tensors['in2'] = new Tensor('in2', [2], 'int32', false, true, new Int32Array([1, 1]));
      g.tensors['in3'] = new Tensor('in3', [1], 'float32', false, true, new Float32Array([0.0]));
    });
  });

  it('Shape error', async () => {
    const n = new Node('Shape', ['in1'], ['out']);
    await expect(compileNode(n)).rejects.toThrow('Shape should be evaluated on CPU statically');
  });

  it('Squeeze / Unsqueeze / Tile errors', async () => {
    await expect(compileNode(new Node('Squeeze', ['in1'], ['out']))).rejects.toThrow(
      'Squeeze implemented via reshape mapping',
    );
    await expect(compileNode(new Node('Unsqueeze', ['in1'], ['out']))).rejects.toThrow(
      'Unsqueeze implemented via reshape mapping',
    );
    await expect(compileNode(new Node('Tile', ['in1'], ['out']))).rejects.toThrow(
      'Tile requires expanding/concatenating',
    );
  });

  it('Split with explicit tensor array', async () => {
    const n = new Node('Split', ['in1', 'in2'], ['out', 'out2']);
    await compileNode(n, (g) => {
      g.outputs.push({ name: 'out2', shape: [1], id: 'o2', dtype: 'float32' } as any);
      g.tensors['in2'] = new Tensor('in2', [2], 'int32', false, true, new Int32Array([1, 1]));
    });
  });
});
