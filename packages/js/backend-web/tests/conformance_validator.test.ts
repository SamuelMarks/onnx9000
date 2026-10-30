import { Attribute, Graph, Node } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { WebNNLayoutValidator } from '../src/providers/webnn/conformance_validator.js';

describe('WebNNLayoutValidator', () => {
  it('should validate layouts', () => {
    const graph = new Graph('test');
    const node = new Node(
      'Conv',
      [],
      [],
      { layout: new Attribute('layout', 'STRING', 'nchw') },
      'c1',
    );
    graph.nodes.push(node);
    expect(WebNNLayoutValidator.validateLayouts(graph)).toBe(true);

    node.attributes.layout = new Attribute('layout', 'STRING', 'invalid');
    expect(() => WebNNLayoutValidator.validateLayouts(graph)).toThrow();
  });
});
