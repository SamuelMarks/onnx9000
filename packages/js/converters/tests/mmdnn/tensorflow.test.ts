import { describe, it, expect } from 'vitest';
import { parsePbtxt } from '../../src/mmdnn/tensorflow/parser.js';
import { TFMapper } from '../../src/mmdnn/tensorflow/mapper.js';
import { Graph } from '@onnx9000/core';

describe('TensorFlow Parser and Mapper', () => {
  it('parses basic nodes, attributes, and maps them', () => {
    const pbtxt = `
node {
  name: "input_node"
  op: "Placeholder"
  attr {
    key: "dtype"
    value { type: DT_FLOAT }
  }
}
node {
  name: "weights"
  op: "Const"
  attr {
    key: "value"
    value { tensor { dtype: DT_FLOAT tensor_shape { dim { size: 32 } dim { size: 3 } dim { size: 3 } dim { size: 3 } } } }
  }
}
node {
  name: "conv"
  op: "Conv2D"
  input: "input_node"
  input: "weights"
  attr {
    key: "strides"
    value { list { i: 1 i: 2 i: 2 i: 1 } }
  }
  attr {
    key: "padding"
    value { s: "SAME" }
  }
}
node {
  name: "relu"
  op: "Relu6"
  input: "conv"
}
node {
  name: "add"
  op: "Add"
  input: "relu"
  input: "relu"
  attr {
    key: "i_val"
    value { i: 42 }
  }
  attr {
    key: "f_val"
    value { f: 3.14 }
  }
  attr {
    key: "shape_val"
    value { shape { dim { size: 10 } } }
  }
}
    `;

    const parsed = parsePbtxt(pbtxt);
    expect(parsed.node.length).toBe(5);

    const graph = new Graph('test_graph');
    graph.initializers = [];
    graph.tensors = {};

    const mapper = new TFMapper();
    const nodes = [];
    for (const node of parsed.node) {
      nodes.push(...mapper.map(node, graph));
    }

    // "Placeholder" gets mapped to Identity
    expect(nodes[0].opType).toBe('Identity');

    // "Const" doesn't produce a node, it becomes an initializer
    expect(graph.initializers).toContain('weights');
    expect(graph.tensors['weights']).toBeDefined();
    expect(graph.tensors['weights'].shape).toEqual([32, 3, 3, 3]);

    // "Conv2D" maps to Conv
    const convNode = nodes[1];
    expect(convNode.opType).toBe('Conv');
    expect(convNode.attributes['strides'].value).toEqual([2, 2]);
    expect(convNode.attributes['auto_pad'].value).toEqual('SAME_UPPER');

    // "Relu6" maps to Relu
    expect(nodes[2].opType).toBe('Relu');

    // "Add" maps to Add and has standard attributes
    const addNode = nodes[3];
    expect(addNode.opType).toBe('Add');
    expect(addNode.attributes['i_val'].value).toBe(42);
    expect(addNode.attributes['f_val'].value).toBeCloseTo(3.14);
    expect(addNode.attributes['shape_val'].value).toEqual([10]);
  });

  it('handles empty and partial strings safely', () => {
    const pbtxt = `
node {
  name: "bad_attr"
  op: "Bad"
  attr {
    key: "val"
    value { }
  }
}
    `;
    const parsed = parsePbtxt(pbtxt);
    expect(parsed.node.length).toBe(1);

    const mapper = new TFMapper();
    const graph = new Graph('test');
    graph.initializers = [];
    graph.tensors = {};

    const nodes = mapper.map(parsed.node[0], graph);
    expect(nodes.length).toBe(1);
    expect(nodes[0].opType).toBe('Bad');
  });

  it('ignores malformed lines gracefully', () => {
    const pbtxt = `
      invalid text here
      node {
        name: "test"
      }
      attr {
      }
    `;
    const parsed = parsePbtxt(pbtxt);
    expect(parsed.node.length).toBe(1);
    expect(parsed.node[0].name).toBe('test');
  });

  it('handles Const without tensor shape gracefully', () => {
    const pbtxt = `
node {
  name: "weights_no_shape"
  op: "Const"
  attr {
    key: "value"
    value { tensor { dtype: DT_FLOAT } }
  }
}
node {
  name: "weights_no_tensor"
  op: "Const"
  attr {
    key: "value"
    value { }
  }
}
    `;
    const parsed = parsePbtxt(pbtxt);
    expect(parsed.node.length).toBe(2);

    const mapper = new TFMapper();
    const graph = new Graph('test');
    graph.initializers = [];
    graph.tensors = {};

    mapper.map(parsed.node[0], graph);
    mapper.map(parsed.node[1], graph);

    expect(graph.initializers).toContain('weights_no_shape');
    expect(graph.initializers).toContain('weights_no_tensor');
    expect(graph.tensors['weights_no_shape'].shape).toEqual([1]);
    expect(graph.tensors['weights_no_tensor'].shape).toEqual([1]);
  });
});
