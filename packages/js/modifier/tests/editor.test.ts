import { describe, it, expect, beforeEach, vi } from 'vitest';
import { Graph, Node } from '@onnx9000/core';
import { GraphMutator } from '../src/GraphMutator.js';
import { GraphEditor } from '../src/ui/editor.js';

describe('GraphEditor UI State', () => {
  let graph: Graph;
  let mutator: GraphMutator;
  let editor: GraphEditor;

  beforeEach(() => {
    graph = new Graph('Test');
    mutator = new GraphMutator(graph);
    editor = new GraphEditor(graph, mutator);
  });

  it('duplicateNode adds a copy fallback name', () => {
    const n1 = mutator.addNode('Op1', ['A'], ['B']);
    n1.name = '';
    editor.duplicateNode(n1.id);
    expect(graph.nodes.length).toBe(2);
  });

  it('selectNode handles single and multi select', () => {
    const cb = vi.fn();
    editor.onSelectionChange = cb;

    editor.selectNode('node1', false);
    expect(editor.selectedNodeIds.has('node1')).toBe(true);
    expect(cb).toHaveBeenCalledWith([{ type: 'node', id: 'node1' }]);

    editor.selectNode('node2', false);
    expect(editor.selectedNodeIds.has('node1')).toBe(false);
    expect(editor.selectedNodeIds.has('node2')).toBe(true);

    editor.selectNode('node3', true);
    expect(editor.selectedNodeIds.has('node2')).toBe(true);
    expect(editor.selectedNodeIds.has('node3')).toBe(true);
  });

  it('selectEdge handles single and multi select', () => {
    editor.selectEdge('edge1', false);
    expect(editor.selectedEdges.has('edge1')).toBe(true);

    editor.selectEdge('edge2', true);
    expect(editor.selectedEdges.size).toBe(2);

    editor.clearSelection();
    expect(editor.selectedEdges.size).toBe(0);
  });

  it('deleteSelection removes nodes and clears edges', () => {
    const n1 = mutator.addNode('Op1', ['A'], ['B'], {}, 'Node1');
    const n2 = mutator.addNode('Op2', ['B'], ['C'], {}, 'Node2');

    editor.selectNode(n1.id, false);
    editor.deleteSelection();

    expect(graph.nodes.length).toBe(1);
    expect(editor.selectedNodeIds.size).toBe(0);

    editor.selectEdge('C'); // C is output of n2
    // Since C is not an input to anything, this does nothing but clears selection
    editor.deleteSelection();
    expect(editor.selectedEdges.size).toBe(0);

    // Now test deleting an edge that IS an input
    mutator.addNode('Op3', ['C'], ['D'], {}, 'Node3');
    editor.selectEdge('C');
    editor.deleteSelection();
    // The input to Op3 should be cleared
    const n3 = graph.nodes.find((n) => n.name === 'Node3')!;
    expect(n3.inputs[0]).toBe('');

    mutator.undo();
    expect(n3.inputs[0]).toBe('C');
  });

  it('disconnectNode clears inputs', () => {
    const n1 = mutator.addNode('Op1', ['A', 'B'], ['C'], {}, 'Node1');
    editor.disconnectNode(n1.id);
    expect(n1.inputs).toEqual(['', '']);

    mutator.undo();
    expect(n1.inputs).toEqual(['A', 'B']);

    // safe exit on non-existent
    editor.disconnectNode('nonexistent');
  });

  it('duplicateNode adds a copy', () => {
    const n1 = mutator.addNode('Op1', ['A'], ['B'], {}, 'Node1');
    editor.duplicateNode(n1.id);

    expect(graph.nodes.length).toBe(2);
    const dup = graph.nodes[1]!;
    expect(dup.opType).toBe('Op1');
    expect(dup.outputs).toEqual(['B_dup']);

    // safe exit on non-existent
    editor.duplicateNode('nonexistent');
  });

  it('connectPorts establishes an edge', () => {
    const n1 = mutator.addNode('Op1', [''], ['B'], {}, 'Node1');
    editor.connectPorts('NewEdge', n1.id, 0);
    expect(n1.inputs[0]).toBe('NewEdge');

    mutator.undo();
    expect(n1.inputs[0]).toBe('');

    // safe exit
    editor.connectPorts('X', 'nonexistent', 0);
  });

  it('201. snapToGrid sets snapped attribute', () => {
    const node = mutator.addNode('Op', [], [], {}, 'SnapNode');
    editor.selectNode(node.id);
    editor.snapToGrid(25);
    expect(node.attributes['snapped_grid']!.value).toBe(25);
  });

  it('202. alignNodes sets alignment attribute', () => {
    const node = mutator.addNode('Op', [], [], {}, 'AlignNode');
    editor.selectNode(node.id);
    editor.alignNodes('Right');
    expect(node.attributes['alignment']!.value).toBe('Right');
  });
});
