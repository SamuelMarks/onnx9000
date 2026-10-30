import { describe, expect, it } from 'vitest';
import { Graph, ValueInfo } from '../src/ir/graph.js';
import { Node } from '../src/ir/node.js';
import { serializeModelProto } from '../src/parser/onnx_writer.js';

describe('onnx_writer', () => {
  it('should serialize model proto', () => {
    const g = new Graph('test');
    g.nodes.push(new Node('Add', ['a', 'b'], ['c'], {}, 'n1'));
    g.inputs.push(new ValueInfo('a', [1], 'float32'));
    g.outputs.push(new ValueInfo('c', [1], 'float32'));

    const bytes = serializeModelProto(g);
    expect(bytes).toBeDefined();
    expect(bytes.length).toBeGreaterThan(0);
  });
});
