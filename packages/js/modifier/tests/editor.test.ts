import { Graph } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { GraphMutator } from '../src/GraphMutator.js';
import { GraphEditor } from '../src/ui/editor.js';

describe('GraphEditor', () => {
  it('should select and delete', () => {
    const g = new Graph('test');
    const mutator = new GraphMutator(g);
    const ed = new GraphEditor(g, mutator);

    ed.selectNode('1');
    expect(ed.selectedNodeIds.has('1')).toBe(true);

    ed.deleteSelection();
    expect(ed.selectedNodeIds.has('1')).toBe(false);
  });
});
