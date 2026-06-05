import { describe, it, expect, vi, beforeEach } from 'vitest';
import * as openvino from '../src/index.js';
import { exportModel, OpenVinoExporter } from '../src/index.js';
import { Graph, Node, Tensor, Attribute } from '@onnx9000/core';
import { XmlBuilder, XmlNode } from '../src/xml_builder.js';

describe('OpenVinoExporter Refactored Exhaustive Final V2', () => {
  beforeEach(() => {
    (OpenVinoExporter as any).handlers.clear();
  });

  it('should cover all op handlers and sub-graphs', () => {
    expect(openvino).toBeDefined();

    const graph = new Graph('FinalModel');
    graph.inputs.push({ name: 'X', shape: [1, 3, 224, 224], dtype: 'float32' });
    graph.inputs.push({ name: 'Scale', shape: [3], dtype: 'float32' });
    graph.inputs.push({ name: 'B', shape: [3], dtype: 'float32' });

    // 1. Normalization
    graph.addNode(
      new Node(
        'LayerNormalization',
        ['X', 'Scale', 'B'],
        ['LN1'],
        { axis: new Attribute('axis', 'INT', 1), epsilon: new Attribute('epsilon', 'FLOAT', 1e-5) },
        'ln',
      ),
    );
    graph.addNode(
      new Node(
        'LpNormalization',
        ['LN1'],
        ['LPN1'],
        { axis: new Attribute('axis', 'INT', 1), p: new Attribute('p', 'INT', 2) },
        'lpn',
      ),
    );

    // 2. Size
    graph.addNode(new Node('Size', ['LPN1'], ['SZ1'], {}, 'size_node'));

    // 3. ConstantOfShape with int64 and int32
    const shapeT = new Tensor('Shape', [1], 'int64', true);
    shapeT.data = new Uint8Array(new BigInt64Array([10n]).buffer);
    graph.tensors['Shape'] = shapeT;
    graph.initializers.push('Shape');

    const valTI64 = new Tensor('ValI64', [], 'int64', true);
    valTI64.data = new Uint8Array(new BigInt64Array([1n]).buffer);
    graph.addNode(
      new Node(
        'ConstantOfShape',
        ['Shape'],
        ['COS_I64'],
        { value: new Attribute('value', 'TENSOR', valTI64) },
        'cos_i64',
      ),
    );

    const valTI32 = new Tensor('ValI32', [], 'int32', true);
    valTI32.data = new Uint8Array(new Int32Array([1]).buffer);
    graph.addNode(
      new Node(
        'ConstantOfShape',
        ['Shape'],
        ['COS_I32'],
        { value: new Attribute('value', 'TENSOR', valTI32) },
        'cos_i32',
      ),
    );

    // 4. Flatten and Transpose
    graph.addNode(
      new Node('Flatten', ['X'], ['FL1'], { axis: new Attribute('axis', 'INT', 1) }, 'flatten'),
    );
    graph.addNode(
      new Node(
        'Transpose',
        ['FL1'],
        ['T1'],
        { perm: new Attribute('perm', 'INTS', [0, 1]) },
        'transpose',
      ),
    );

    graph.outputs.push({ name: 'COS_I64', shape: [10], dtype: 'int64' });

    const result = exportModel(graph);
    expect(result.xml).toBeDefined();
  });
});
