// @vitest-environment jsdom
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { Graph, Node } from '@onnx9000/core';
import { GraphMutator } from '../src/GraphMutator.js';
import { PropertiesPanel } from '../src/components/properties.js';

describe('PropertiesPanel', () => {
  let container: HTMLElement;
  let mutator: GraphMutator;
  let graph: Graph;
  let panel: PropertiesPanel;

  beforeEach(() => {
    graph = new Graph('Test');
    mutator = new GraphMutator(graph);
    container = document.createElement('div');
    panel = new PropertiesPanel(container, mutator);
  });

  it('handles property changes WITH calling mutator for string', () => {
    const node = mutator.addNode(
      'Op1',
      ['A'],
      ['B'],
      { attr3: { name: 'attr3', type: 'STRING', value: 'hello' } },
      'Node1',
    );
    panel.renderNode(node, graph);
    const inputs = container.querySelectorAll('input');
    const strInput = Array.from(inputs).find((i) => i.value === 'hello')!;
    strInput.value = 'world';
    strInput.dispatchEvent(new Event('change'));
    expect(node.attributes['attr3']!.value).toEqual('world');
  });

  it('handles empty property changes WITH calling mutator for arrays no-op if not array', () => {
    const node = mutator.addNode(
      'Op1',
      ['A'],
      ['B'],
      { attr4: { name: 'attr4', type: 'INTS', value: [1, 2] } },
      'Node1',
    );
    panel.renderNode(node, graph);
    const inputs = container.querySelectorAll('input');
    const arrInput = Array.from(inputs).find((i) => i.value === '[1,2]')!;
    arrInput.value = '1';
    arrInput.dispatchEvent(new Event('change'));
    expect(node.attributes['attr4']!.value).toEqual([1, 2]);
  });

  it('handles empty property changes WITH calling mutator for arrays', () => {
    const node = mutator.addNode(
      'Op1',
      ['A'],
      ['B'],
      { attr4: { name: 'attr4', type: 'INTS', value: [1, 2] } },
      'Node1',
    );
    panel.renderNode(node, graph);
    const inputs = container.querySelectorAll('input');
    const arrInput = Array.from(inputs).find((i) => i.value === '[1,2]')!;
    arrInput.value = '[3]';
    arrInput.dispatchEvent(new Event('change'));
    expect(node.attributes['attr4']!.value).toEqual([3]);
  });

  it('handles property changes with calling mutator for primitives', () => {
    const node = mutator.addNode(
      'Op1',
      ['A'],
      ['B'],
      {
        attr1: { name: 'attr1', type: 'INT', value: 1 },
        attr2: { name: 'attr2', type: 'FLOAT', value: 1.0 },
      },
      'Node1',
    );
    panel.renderNode(node, graph);
    const inputs = container.querySelectorAll('input');
    const intInput = Array.from(inputs).find((i) => i.value === '1')!;
    intInput.value = '2';
    intInput.dispatchEvent(new Event('change'));
    expect(node.attributes['attr1']!.value).toEqual(2);

    // find the float input which is also 1
    const floatInput = Array.from(inputs).find((i) => i.value === '1')!;
    floatInput.value = '3.0';
    floatInput.dispatchEvent(new Event('change'));
    expect(node.attributes['attr2']!.value).toEqual(3.0);
  });

  it('handles empty property changes without calling mutator', () => {
    const node = mutator.addNode('Op1', ['A'], ['B'], {}, 'Node1');
    panel.renderNode(node, graph);
    const inputs = container.querySelectorAll('input');
    const originalName = node.name;
    inputs[0]!.value = '';
    inputs[0]!.dispatchEvent(new Event('change'));
    expect(node.name).toBe(originalName);

    const originalOp = node.opType;
    inputs[1]!.value = '';
    inputs[1]!.dispatchEvent(new Event('change'));
    expect(node.opType).toBe(originalOp);
  });

  it('handles empty edge inputs safely', () => {
    const node1 = mutator.addNode('Op1', [''], ['B'], {}, 'Node1');
    node1.name = '';
    panel.renderNode(node1, graph);
  });
  it('handles empty edge outputs safely', () => {
    const node1 = mutator.addNode('Op1', ['A'], [''], {}, 'Node1');
    node1.name = '';
    panel.renderNode(node1, graph);
  });

  it('handles empty edge input values gracefully', () => {
    const node = mutator.addNode('Op1', ['A'], ['B'], {}, 'Node1');
    node.name = '';
    panel.renderNode(node, graph);
    const inputs = container.querySelectorAll('input');
    inputs[0]!.value = '';
    inputs[0]!.dispatchEvent(new Event('change'));
    expect(node.name).toBe('');
    inputs[1]!.value = '';
    inputs[1]!.dispatchEvent(new Event('change'));
    expect(node.opType).toBe('Op1');
  });

  it('handles edge with empty node names', () => {
    const node1 = mutator.addNode('Op1', ['A'], ['B'], {}, '');
    const node2 = mutator.addNode('Op2', ['B'], ['C'], {}, '');
    node1.name = '';
    node2.name = '';
    panel.renderEdge('B', graph);
    expect(container.innerHTML).toContain(node1.id);
    expect(container.innerHTML).toContain(node2.id);
  });

  it('renders node properties and supports editing', () => {
    const node = mutator.addNode(
      'Op1',
      ['A'],
      ['B'],
      {
        attr1: { name: 'attr1', type: 'INT', value: 1 },
        attr2: { name: 'attr2', type: 'FLOAT', value: 1.0 },
        attr3: { name: 'attr3', type: 'STRING', value: 'hello' },
        attr4: { name: 'attr4', type: 'INTS', value: [1, 2] },
      },
      'Node1',
    );

    panel.renderNode(node, graph);
    expect(container.innerHTML).toContain('Op1'); // H3 replaced Node Properties with OpType
    expect(container.innerHTML).toContain('Node1');
    expect(container.innerHTML).toContain('OpType');

    // Find the inputs and trigger changes
    const inputs = container.querySelectorAll('input');
    expect(inputs.length).toBeGreaterThan(0);

    // change name
    inputs[0]!.value = 'Node1_New';
    inputs[0]!.dispatchEvent(new Event('change'));
    expect(graph.nodes[0]!.name).toBe('Node1_New');

    // change optype
    inputs[1]!.value = 'Op2';
    inputs[1]!.dispatchEvent(new Event('change'));
    expect(graph.nodes[0]!.opType).toBe('Op2');

    // change int attr
    const intInput = Array.from(inputs).find((i) => i.value === '1')!;
    intInput.value = '2';
    intInput.dispatchEvent(new Event('change'));
    expect(graph.nodes[0]!.attributes['attr1']!.value).toBe(2);

    // change float attr
    const floatInput = Array.from(inputs).find((i) => i.value === '1')!; // String(1.0) is 1
    floatInput.value = '3.14';
    floatInput.dispatchEvent(new Event('change'));
    expect(graph.nodes[0]!.attributes['attr2']!.value).toBe(3.14);

    // change string attr
    const strInput = Array.from(inputs).find((i) => i.value === 'hello')!;
    strInput.value = 'world';
    strInput.dispatchEvent(new Event('change'));
    expect(graph.nodes[0]!.attributes['attr3']!.value).toBe('world');

    // change array attr
    const arrInput = Array.from(inputs).find((i) => i.value === '[1,2]')!;
    arrInput.value = '[3,4]';
    arrInput.dispatchEvent(new Event('change'));
    expect(graph.nodes[0]!.attributes['attr4']!.value).toEqual([3, 4]);

    // bad JSON parsing should not throw
    arrInput.value = '[bad json]';
    expect(() => arrInput.dispatchEvent(new Event('change'))).not.toThrow();
  });

  it('renders edge properties with no shape info', () => {
    panel.renderEdge('unknown_edge', graph);
    expect(container.innerHTML).toContain('Edge Properties');
    expect(container.innerHTML).toContain('unknown_edge');
    expect(container.innerHTML).toContain('Unknown shape/type');
  });

  it('renders edge properties with shape info, producers, and consumers', () => {
    graph.inputs.push({ name: 'A', shape: [1], dtype: 'float32', id: '' });
    mutator.addNode('Op1', ['A'], ['B'], {}, 'Node1');
    mutator.addNode('Op2', ['B'], ['C'], {}, 'Node2');

    panel.renderEdge('A', graph);
    expect(container.innerHTML).toContain('float32');
    expect(container.innerHTML).toContain('Node1'); // consumer
    expect(container.innerHTML).toContain('None (Input/Initializer)'); // producer

    panel.renderEdge('B', graph);
    expect(container.innerHTML).toContain('Node1'); // producer
    expect(container.innerHTML).toContain('Node2'); // consumer
  });
});
