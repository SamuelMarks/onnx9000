import { describe, it, expect } from 'vitest';
import { WebNNLayoutValidator } from '../../../src/providers/webnn/conformance_validator';
import { Graph, Node, Attribute } from '@onnx9000/core';

describe('WebNNLayoutValidator', () => {
  it('should validate valid layouts', () => {
    const g = new Graph('test');
    const node = new Node('Conv', [], []);
    node.name = 'conv1';
    node.attributes['layout'] = new Attribute('layout', 'string', 'NCHW');
    g.nodes.push(node);

    expect(WebNNLayoutValidator.validateLayouts(g)).toBe(true);
  });

  it('should throw on missing layout', () => {
    const g = new Graph('test');
    const node = new Node('Conv', [], []);
    node.name = 'conv1';
    g.nodes.push(node);

    expect(() => WebNNLayoutValidator.validateLayouts(g)).toThrow(/missing a layout attribute/);
  });

  it('should throw on invalid layout', () => {
    const g = new Graph('test');
    const node = new Node('Conv', [], []);
    node.name = 'conv1';
    node.attributes['layout'] = new Attribute('layout', 'string', 'BCHW');
    g.nodes.push(node);

    expect(() => WebNNLayoutValidator.validateLayouts(g)).toThrow(/invalid layout/);
  });
});
