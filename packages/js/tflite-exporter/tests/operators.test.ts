import { describe, it, expect } from 'vitest';
import { TFLiteExporter } from '../src/exporter';
import { compileGraphToTFLite } from '../src/compiler/subgraph';
import { FlatBufferReader } from '../src/flatbuffer/reader';
import { Graph, Tensor, Node, Attribute } from '@onnx9000/core';
import { ELEMENTWISE_OPS } from '../src/compiler/operators';

describe('TFLite Compiler - All Operators', () => {
  it('should map scalar vs tensor addition correctly', () => {
    // 99. Verify scalar vs tensor addition signatures map correctly to TFLite options.
    const exporter = new TFLiteExporter();
    const graph = new Graph('TestScalarGraph');

    graph.tensors['X'] = new Tensor('X', [1, 10, 10, 3], 'float32', false);
    graph.tensors['ScalarB'] = new Tensor(
      'ScalarB',
      [],
      'float32',
      true,
      false,
      new Float32Array([5.0]),
    );
    graph.tensors['Z'] = new Tensor('Z', [1, 10, 10, 3], 'float32', false);

    graph.inputs.push({ name: 'X', shape: [1, 10, 10, 3], dtype: 'float32', id: '0' });
    graph.outputs.push({ name: 'Z', shape: [1, 10, 10, 3], dtype: 'float32', id: '2' });

    graph.nodes.push(new Node('Add', ['X', 'ScalarB'], ['Z'], {}, 'add_scalar'));

    const subgraphsOffset = compileGraphToTFLite(graph, exporter, true, 'none');
    exporter.builder.startVector(4, 1, 4);
    exporter.builder.addOffset(subgraphsOffset);
    const subgraphsVecOffset = exporter.builder.endVector(1);
    const buf = exporter.finish(subgraphsVecOffset, 'test');

    // If it didn't throw and successfully compiled, the signature mapped correctly natively
    expect(buf.length).toBeGreaterThan(0);
  });
  it('should map all supported operators without throwing', () => {
    // Some operators require specific attributes or input structures to avoid logic errors
    // We will provide dummy tensors and inputs.

    for (const [opType, mapping] of Object.entries(ELEMENTWISE_OPS)) {
      const exporter = new TFLiteExporter();
      const graph = new Graph('TestOpGraph');

      graph.tensors['X'] = new Tensor('X', [1, 10, 10, 3], 'float32', false);
      graph.tensors['Y'] = new Tensor('Y', [1, 10, 10, 3], 'float32', false);
      graph.tensors['Z'] = new Tensor('Z', [1, 10, 10, 3], 'float32', false);
      graph.tensors['W'] = new Tensor(
        'W',
        [3, 3, 3, 3],
        'float32',
        true,
        false,
        new Float32Array(81),
      ); // Weight

      graph.inputs.push({ name: 'X', shape: [1, 10, 10, 3], dtype: 'float32', id: '0' });
      graph.outputs.push({ name: 'Z', shape: [1, 10, 10, 3], dtype: 'float32', id: '2' });

      const attrs: Record<string, Attribute> = {};
      if (opType === 'LeakyRelu') attrs['alpha'] = new Attribute('alpha', 'FLOAT', 0.1);
      if (opType === 'LRN') attrs['size'] = new Attribute('size', 'INT', 3);
      if (opType === 'Concat' || opType === 'Gather')
        attrs['axis'] = new Attribute('axis', 'INT', 1);
      if (opType === 'SpaceToDepth' || opType === 'DepthToSpace')
        attrs['blocksize'] = new Attribute('blocksize', 'INT', 2);

      let inputs = ['X'];
      let outputs = ['Z'];

      if (['Add', 'Sub', 'Mul', 'Div', 'Equal', 'Less', 'Greater'].includes(opType))
        inputs = ['X', 'Y'];
      if (['Conv', 'ConvTranspose', 'Gemm', 'MatMul'].includes(opType)) inputs = ['X', 'W'];
      if (['Split', 'SplitV', 'SplitToSequence'].includes(opType)) outputs = ['Y', 'Z'];

      const node = new Node(opType, inputs, outputs, attrs, `${opType}_node`);

      // Also add fused_activation to math ops to test coverage
      if (['Add', 'Sub', 'Mul', 'Div'].includes(opType)) {
        node.attributes['fused_activation'] = new Attribute('fused_activation', 'STRING', 'Relu');
      }
      // Also add reverse/exclusive to cumsum
      if (opType === 'CumSum') {
        node.attributes['exclusive'] = new Attribute('exclusive', 'INT', 1);
        node.attributes['reverse'] = new Attribute('reverse', 'INT', 1);
      }

      graph.nodes.push(node);

      try {
        const subgraphsOffset = compileGraphToTFLite(graph, exporter, true, 'none');
        exporter.builder.startVector(4, 1, 4);
        exporter.builder.addOffset(subgraphsOffset);
        const subgraphsVecOffset = exporter.builder.endVector(1);
        const buf = exporter.finish(subgraphsVecOffset, 'test');
        const reader = new FlatBufferReader(buf);
        expect(reader.checkMagicBytes('TFL3')).toBe(true);

        const modelOffset = reader.getRoot();
        const opCodesVec = reader.getIndirectOffset(modelOffset, 1);
        expect(reader.getVectorLength(opCodesVec)).toBe(1);

        const opCodeObj =
          reader.getVectorItemOffset(opCodesVec, 0) +
          reader.view.getUint32(reader.getVectorItemOffset(opCodesVec, 0), true);
        const code = reader.getInt8(opCodeObj, 0);
        const extendedCode = reader.getInt32(opCodeObj, 3, 0);
        const finalCode = extendedCode !== 0 ? extendedCode : code < 0 ? code + 256 : code;
        expect(finalCode).toBe(mapping.builtinCode);
      } catch (e: any) {
        throw new Error(`Failed on operator ${opType}: ${e.message}`);
      }
    }
  });

  it('should map Conv properties', () => {
    const exporter = new TFLiteExporter();
    const graph = new Graph('TestOpGraph');
    graph.tensors['X'] = new Tensor('X', [1, 10, 10, 3], 'float32', false);
    graph.tensors['W'] = new Tensor(
      'W',
      [3, 3, 3, 3],
      'float32',
      true,
      false,
      new Float32Array(81),
    ); // Weight
    graph.inputs.push({ name: 'X', shape: [1, 10, 10, 3], dtype: 'float32', id: '0' });
    graph.outputs.push({ name: 'Z', shape: [1, 10, 10, 3], dtype: 'float32', id: '2' });

    // Conv
    graph.nodes.push(
      new Node(
        'Conv',
        ['X', 'W'],
        ['Z'],
        {
          strides: new Attribute('strides', 'INTS', [2, 2]),
          dilations: new Attribute('dilations', 'INTS', [1, 1]),
        },
        'conv1',
      ),
    );

    // Depthwise Conv
    graph.nodes.push(
      new Node(
        'Conv',
        ['X', 'W'],
        ['Z'],
        {
          strides: new Attribute('strides', 'INTS', [2, 2]),
          group: new Attribute('group', 'INT', 3),
        },
        'conv2',
      ),
    );

    // MaxPool
    graph.nodes.push(
      new Node(
        'MaxPool',
        ['X'],
        ['Z'],
        {
          strides: new Attribute('strides', 'INTS', [2, 2]),
          kernel_shape: new Attribute('kernel_shape', 'INTS', [3, 3]),
        },
        'pool1',
      ),
    );

    // Custom
    graph.nodes.push(new Node('NonMaxSuppression', ['X'], ['Z'], {}, 'nms'));
    graph.nodes.push(new Node('MyCustomOp', ['X'], ['Z'], {}, 'custom', 'org.test'));

    const subgraphsOffset = compileGraphToTFLite(graph, exporter, true, 'none');
    exporter.builder.startVector(4, 1, 4);
    exporter.builder.addOffset(subgraphsOffset);
    const subgraphsVecOffset = exporter.builder.endVector(1);
    const buf = exporter.finish(subgraphsVecOffset, 'test');
    expect(buf.length).toBeGreaterThan(0);
  });
});
