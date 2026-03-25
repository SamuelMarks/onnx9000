import { describe, it, expect } from 'vitest';
import { parsePbtxt } from '../../../src/mmdnn/tensorflow/parser.js';

describe('parsePbtxt', () => {
  it('should parse a basic node', () => {
    const pbtxt = `
node {
  name: "input_1"
  op: "Placeholder"
  attr {
    key: "dtype"
    value { type: DT_FLOAT }
  }
  attr {
    key: "shape"
    value { shape { dim { size: -1 } dim { size: 224 } dim { size: 224 } dim { size: 3 } } }
  }
}
    `;
    const graph = parsePbtxt(pbtxt);
    expect(graph.node.length).toBe(1);
    expect(graph.node[0].name).toBe('input_1');
    expect(graph.node[0].op).toBe('Placeholder');
    expect(graph.node[0].attr['dtype'].type).toBe('DT_FLOAT');
    expect(graph.node[0].attr['shape'].shape).toEqual([-1, 224, 224, 3]);
  });

  it('should parse node with inputs', () => {
    const pbtxt = `
node {
  name: "Conv2D_1"
  op: "Conv2D"
  input: "input_1"
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
    `;
    const graph = parsePbtxt(pbtxt);
    expect(graph.node[0].input).toEqual(['input_1', 'weights']);
    expect(graph.node[0].attr['strides'].list?.i).toEqual([1, 2, 2, 1]);
    expect(graph.node[0].attr['padding'].s).toBe('SAME');
  });

  it('should parse node with tensor attr', () => {
    const pbtxt = `
node {
  name: "weights"
  op: "Const"
  attr {
    key: "value"
    value { tensor { dtype: DT_FLOAT tensor_shape { dim { size: 32 } dim { size: 3 } dim { size: 3 } dim { size: 3 } } } }
  }
}
    `;
    const graph = parsePbtxt(pbtxt);
    expect(graph.node[0].attr['value'].tensor?.dtype).toBe('DT_FLOAT');
    expect(graph.node[0].attr['value'].tensor?.shape).toEqual([32, 3, 3, 3]);
  });

  it('should parse integer and float attrs', () => {
    const pbtxt = `
node {
  name: "test"
  attr {
    key: "int_val"
    value { i: 42 }
  }
  attr {
    key: "float_val"
    value { f: 3.14 }
  }
}
    `;
    const graph = parsePbtxt(pbtxt);
    expect(graph.node[0].attr['int_val'].i).toBe(42);
    expect(graph.node[0].attr['float_val'].f).toBe(3.14);
  });
});
