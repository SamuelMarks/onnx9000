import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Graph, Node, Tensor, Attribute } from '@onnx9000/core';
import { WebNNCompiler } from '../src/providers/webnn/compiler.js';

describe('WebNNCompiler specific lines coverage', () => {
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
      transpose() {
        return {};
      }
    };
    builder = new (globalThis as any).MLGraphBuilder();
  });

  const compileNode = async (node: Node, setupGraph?: (g: Graph) => void) => {
    const g = new Graph('test');
    g.inputs.push({ name: 'in1', shape: [1], id: 'i1', dtype: 'float32' });
    g.inputs.push({ name: 'in2', shape: [1], id: 'i2', dtype: 'float32' });
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

  it('covers missing dtypes and addConstant return', async () => {
    await compileNode(new Node('Add', ['in1', 'in2'], ['out']), (g) => {
      g.tensors['empty_init'] = new Tensor('empty_init', [1], 'float32', false, true, undefined);
      g.initializers.push('empty_init');

      g.tensors['t_f16'] = new Tensor('t_f16', [1], 'float16', false, true, new Uint16Array([0]));
      g.initializers.push('t_f16');

      g.tensors['t_i8'] = new Tensor('t_i8', [1], 'int8', false, true, new Int8Array([0]));
      g.initializers.push('t_i8');

      g.tensors['t_u8'] = new Tensor('t_u8', [1], 'uint8', false, true, new Uint8Array([0]));
      g.initializers.push('t_u8');
    });
  });

  it('covers missing ops: Transpose, Concat, Split, Expand, Gather, Cast', async () => {
    const nTranspose = new Node('Transpose', ['in1'], ['out']);
    await compileNode(nTranspose);

    const nConcat = new Node('Concat', ['in1', 'in2'], ['out']);
    nConcat.attributes['axis'] = new Attribute('axis', 'INT', 1);
    await compileNode(nConcat);

    const nSplit = new Node('Split', ['in1'], ['out', 'out2']);
    await compileNode(nSplit, (g) => {
      g.outputs.push({ name: 'out2', shape: [1], id: 'o2', dtype: 'float32' } as any);
    });

    const nExpand = new Node('Expand', ['in1', 'in2'], ['out']);
    await compileNode(nExpand, (g) => {
      g.tensors['in2'] = new Tensor('in2', [1], 'int32', false, true, new Int32Array([1]));
    });

    const nGather = new Node('Gather', ['in1', 'in2'], ['out']);
    await compileNode(nGather);

    const nCast = new Node('Cast', ['in1'], ['out']);
    nCast.attributes['to'] = new Attribute('to', 'INT', 10); // float16
    await compileNode(nCast);
  });

  it('covers missing attribute extractors', async () => {
    const n = new Node('Transpose', ['in1'], ['out']);
    await compileNode(n);

    const n2 = new Node('BatchNormalization', ['in1', 'in2', 'in3', 'in4', 'in5'], ['out']);
    await compileNode(n2, (g) => {
      g.inputs.push({ name: 'in3', shape: [1], id: 'i3', dtype: 'float32' });
      g.inputs.push({ name: 'in4', shape: [1], id: 'i4', dtype: 'float32' });
      g.inputs.push({ name: 'in5', shape: [1], id: 'i5', dtype: 'float32' });
    });
  });

  it('covers LayerNormalization decomp fallback without scale and bias', async () => {
    builder.layerNormalization = () => {
      throw new Error('Not supported');
    };
    const n = new Node('LayerNormalization', ['in1'], ['out']);
    await compileNode(n);
  });

  it('covers LayerNormalization decomp fallback WITH scale ONLY', async () => {
    builder.layerNormalization = () => {
      throw new Error('Not supported');
    };
    const n = new Node('LayerNormalization', ['in1', 'in2'], ['out']);
    await compileNode(n);
  });

  it('covers QuantizeLinear throw', async () => {
    builder.quantizeLinear = undefined;
    const n = new Node('QuantizeLinear', ['in1', 'in2'], ['out']);
    await expect(compileNode(n)).rejects.toThrow(
      'QuantizeLinear is not supported in this WebNN implementation.',
    );
  });

  it('covers QuantizeLinear without zero point', async () => {
    builder.quantizeLinear = vi.fn().mockReturnValue({});
    const n = new Node('QuantizeLinear', ['in1', 'in2'], ['out']);
    await compileNode(n);
  });

  it('covers DequantizeLinear without zero point (native)', async () => {
    builder.dequantizeLinear = vi.fn().mockReturnValue({});
    const n = new Node('DequantizeLinear', ['in1', 'in2'], ['out']);
    await compileNode(n);
  });
});
