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
      clamp() {
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

  it('covers Slice with and without axes/steps', async () => {
    const nSlice = new Node('Slice', ['in1', 'starts', 'ends'], ['out']);
    await compileNode(nSlice, (g) => {
      g.tensors['starts'] = new Tensor('starts', [1], 'int32', false, true, new Int32Array([0]));
      g.tensors['ends'] = new Tensor('ends', [1], 'int32', false, true, new Int32Array([1]));
      g.initializers.push('starts', 'ends');
    });

    const nSlice2 = new Node('Slice', ['in1', 'starts', 'ends', 'axes', 'steps'], ['out']);
    await compileNode(nSlice2, (g) => {
      g.tensors['starts'] = new Tensor('starts', [1], 'int32', false, true, new Int32Array([0]));
      g.tensors['ends'] = new Tensor('ends', [1], 'int32', false, true, new Int32Array([1]));
      g.tensors['axes'] = new Tensor('axes', [1], 'int32', false, true, new Int32Array([0]));
      g.tensors['steps'] = new Tensor('steps', [1], 'int32', false, true, new Int32Array([1]));
      g.initializers.push('starts', 'ends', 'axes', 'steps');
    });

    const nSliceBad = new Node('Slice', ['in1', 'bad_starts', 'ends'], ['out']);
    await expect(
      compileNode(nSliceBad, (g) => {
        g.inputs.push({ name: 'bad_starts', shape: [1], id: 'i9', dtype: 'int32' });
        g.tensors['ends'] = new Tensor('ends', [1], 'int32', false, true, new Int32Array([1]));
        g.initializers.push('ends');
      }),
    ).rejects.toThrow('Slice requires constant starts and ends in WebNN v1');
  });

  it('covers Pad mode branches and missing pads', async () => {
    const nPadReflect = new Node('Pad', ['in1', 'pads'], ['out']);
    nPadReflect.attributes['mode'] = new Attribute('mode', 'STRING', 'reflect');
    await compileNode(nPadReflect, (g) => {
      g.tensors['pads'] = new Tensor('pads', [2], 'int32', false, true, new Int32Array([1, 1]));
      g.initializers.push('pads');
    });

    const nPadEdge = new Node('Pad', ['in1', 'pads'], ['out']);
    nPadEdge.attributes['mode'] = new Attribute('mode', 'STRING', 'edge');
    await compileNode(nPadEdge, (g) => {
      g.tensors['pads'] = new Tensor('pads', [2], 'int32', false, true, new Int32Array([1, 1]));
      g.initializers.push('pads');
    });

    const nPadBad = new Node('Pad', ['in1'], ['out']);
    await expect(compileNode(nPadBad)).rejects.toThrow('Pad requires constant pads tensor');
  });

  it('covers Reshape without shapeData', async () => {
    const n = new Node('Reshape', ['in1'], ['out']);
    await expect(compileNode(n)).rejects.toThrow(
      'Reshape requires a constant shape tensor in WebNN v1',
    );
  });

  it('covers Softmax with axis', async () => {
    const n = new Node('Softmax', ['in1'], ['out']);
    n.attributes['axis'] = new Attribute('axis', 'INT', 1);
    await compileNode(n);
  });

  it('covers Clip with min and max', async () => {
    const n = new Node('Clip', ['in1', 'min', 'max'], ['out']);
    await compileNode(n, (g) => {
      g.tensors['min'] = new Tensor('min', [], 'float32', false, true, new Float32Array([0]));
      g.tensors['max'] = new Tensor('max', [], 'float32', false, true, new Float32Array([1]));
      g.initializers.push('min', 'max');
    });

    // Line 146: !tensor in extractFloat32TensorData
    const n3 = new Node('Pad', ['in1', 'pads', 'not_found'], ['out3']);
    n3.attributes['mode'] = new Attribute('mode', 'STRING', 'constant');
    await compileNode(n3, (g) => {
      g.tensors['pads'] = new Tensor('pads', [2], 'int32', false, true, new Int32Array([1, 1]));
      g.initializers.push('pads');
      g.inputs.push({ name: 'not_found', shape: [], id: 'm4', dtype: 'float32' } as any);
    });

    // Line 147: !tensor.data in extractFloat32TensorData
    const n4 = new Node('Pad', ['in1', 'pads', 'no_data'], ['out4']);
    n4.attributes['mode'] = new Attribute('mode', 'STRING', 'constant');
    await compileNode(n4, (g) => {
      g.tensors['pads'] = new Tensor('pads', [2], 'int32', false, true, new Int32Array([1, 1]));
      g.initializers.push('pads');
      g.tensors['no_data'] = new Tensor('no_data', [], 'float32', false, false, undefined);
      g.inputs.push({ name: 'no_data', shape: [], id: 'm5', dtype: 'float32' } as any);
    });
  });

  it('covers Gemm with C and empty string input', async () => {
    const n = new Node('Gemm', ['in1', 'in2', 'in3'], ['out']);
    await compileNode(n, (g) => {
      g.inputs.push({ name: 'in3', shape: [1], id: 'i3', dtype: 'float32' } as any);
    });

    // Line 157: name === ''
    const nEmpty = new Node('Gemm', ['in1', 'in2', ''], ['outEmpty']);
    await compileNode(nEmpty);
  });

  it('covers addConstant with dynamic shape', async () => {
    const n = new Node('Add', ['in1', 'dyn'], ['out']);
    await compileNode(n, (g) => {
      g.tensors['dyn'] = new Tensor(
        'dyn',
        ['batch' as any, 2],
        'float32',
        false,
        true,
        new Float32Array([1, 2]),
      );
      g.initializers.push('dyn');
    });
  });

  it('covers Transpose with perm', async () => {
    const n = new Node('Transpose', ['in1'], ['out']);
    n.attributes['perm'] = new Attribute('perm', 'INTS', [1, 0]);
    await compileNode(n);
  });

  it('covers Expand without shapeData', async () => {
    const n = new Node('Expand', ['in1'], ['out']);
    await expect(compileNode(n)).rejects.toThrow(
      'Expand requires a constant shape tensor in WebNN v1',
    );
  });

  it('covers Cast with unknown to type', async () => {
    const nCast = new Node('Cast', ['in1'], ['out']);
    nCast.attributes['to'] = new Attribute('to', 'INT', 9999);
    await compileNode(nCast);

    const nCast2 = new Node('Cast', ['in1'], ['out']);
    // missing 'to' entirely
    await compileNode(nCast2);
  });

  it('covers LayerNormalization fallback with axes', async () => {
    builder.layerNormalization = () => {
      throw new Error('Not supported');
    };
    const n = new Node('LayerNormalization', ['in1'], ['out']);
    n.attributes['axis'] = new Attribute('axis', 'INT', 1);
    await compileNode(n);
  });
});
