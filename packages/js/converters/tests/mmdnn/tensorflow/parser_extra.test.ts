import { describe, it, expect } from 'vitest';
import { parsePbtxt } from '../../../src/mmdnn/tensorflow/parser.js';

describe('parsePbtxt extra', () => {
  it('should parse tensor attribute and list attribute', () => {
    const pbtxt = `
node {
  name: "test"
  op: "Const"
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim { size: 1 }
          dim { size: 10 }
        }
      }
    }
  }
  attr {
    key: "strides"
    value {
      list {
        i: 1
        i: 2
      }
    }
  }
}
    `;
    const graph = parsePbtxt(pbtxt);
    expect(graph.node[0].attr['value'].tensor).toBeDefined();
    expect(graph.node[0].attr['value'].tensor?.shape).toEqual([1, 10]);
    expect(graph.node[0].attr['strides'].list?.i).toEqual([1, 2]);
  });
});
