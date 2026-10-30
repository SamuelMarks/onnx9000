import { Graph, Node } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { renderCustomEditor } from '../src/components/editors/custom_editors.js';
import { GraphMutator } from '../src/GraphMutator.js';

describe('custom_editors', () => {
  it('should render conv editor', () => {
    const container = document.createElement('div');
    const node = new Node('Conv', [], [], {});
    const mutator = new GraphMutator(new Graph('test'));

    expect(renderCustomEditor({ container, node, mutator })).toBe(true);
    expect(container.innerHTML).toContain('Conv Settings');
  });

  it('should not render unknown', () => {
    const container = document.createElement('div');
    const node = new Node('Unknown', [], [], {});
    const mutator = new GraphMutator(new Graph('test'));

    expect(renderCustomEditor({ container, node, mutator })).toBe(false);
  });
});
