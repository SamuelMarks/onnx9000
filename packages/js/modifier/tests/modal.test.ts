// @vitest-environment jsdom
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { Graph } from '@onnx9000/core';
import { GraphMutator } from '../src/GraphMutator.js';
import { AddNodeModal } from '../src/components/modal.js';

describe('AddNodeModal', () => {
  let container: HTMLElement;
  let mutator: GraphMutator;
  let graph: Graph;
  let modal: AddNodeModal;

  beforeEach(() => {
    graph = new Graph('Test');
    mutator = new GraphMutator(graph);
    container = document.createElement('div');
    modal = new AddNodeModal(container, mutator);
  });

  it('renders and adds node', () => {
    modal.show();
    expect(container.style.display).toBe('block');
    expect(container.innerHTML).toContain('Add Node');

    const inputs = container.querySelectorAll('input');
    expect(inputs.length).toBe(2);

    const buttons = container.querySelectorAll('button');
    expect(buttons.length).toBe(2);

    // Test adding
    inputs[0]!.value = 'Relu'; // type
    inputs[1]!.value = 'MyRelu'; // name

    // Click Add
    buttons[0]!.click();
    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0]!.opType).toBe('Relu');
    expect(graph.nodes[0]!.name).toBe('MyRelu');
    expect(container.style.display).toBe('none'); // it hides
  });

  it('does not add if type is empty', () => {
    modal.show();
    const buttons = container.querySelectorAll('button');
    // type is empty by default
    buttons[0]!.click();
    expect(graph.nodes.length).toBe(0);
  });

  it('cancels and hides', () => {
    modal.show();
    expect(container.style.display).toBe('block');
    const buttons = container.querySelectorAll('button');

    // Click Cancel
    buttons[1]!.click();
    expect(container.style.display).toBe('none');
  });
});
