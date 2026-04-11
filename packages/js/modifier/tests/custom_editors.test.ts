// @vitest-environment jsdom
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { Graph, Node } from '@onnx9000/core';
import { GraphMutator } from '../src/GraphMutator.js';
import { renderCustomEditor } from '../src/components/editors/custom_editors.js';
import { PropertiesPanel } from '../src/components/properties.js';

describe('Custom Editors (Phase 10)', () => {
  let graph: Graph;
  let mutator: GraphMutator;
  let container: HTMLElement;
  let panel: PropertiesPanel;

  beforeEach(() => {
    graph = new Graph('Test');
    mutator = new GraphMutator(graph);
    container = document.createElement('div');
    panel = new PropertiesPanel(container, mutator);
  });

  function renderNodeOp(opType: string, attrs: Record<string, Object> = {}) {
    const node = mutator.addNode(opType, ['A'], ['B'], attrs);
    panel.renderNode(node, graph);
    return node;
  }

  it('111. Conv editor', () => {
    const node = renderNodeOp('Conv', {
      strides: { name: 'strides', type: 'INTS', value: [1, 1] },
    });
    expect(container.textContent).toContain('Conv Settings');
    const inputs = container.querySelectorAll('input');
    expect(inputs.length).toBeGreaterThan(3); // NodeName, OpType, + the 4 attr inputs

    // Simulate change on strides array
    const stridesInput = Array.from(inputs).find((i) => i.type === 'text' && i.value === '[1,1]')!;
    stridesInput.value = '[2,2]';
    stridesInput.dispatchEvent(new window.Event('change', { bubbles: true }));
    expect(mutator.graph.nodes[0]!.attributes['strides']!.value).toEqual([2, 2]);

    // Simulate change on group number
    const groupInput = Array.from(inputs).find((i) => i.type === 'number') as HTMLInputElement;
    groupInput.value = '2';
    groupInput.dispatchEvent(new window.Event('change', { bubbles: true }));
    expect(mutator.graph.nodes[0]!.attributes['group']!.value).toBe(2);
  });

  it('112. Gemm editor', () => {
    renderNodeOp('Gemm', {
      transA: { name: 'transA', type: 'INT', value: 1 },
    });
    expect(container.textContent).toContain('Gemm Settings');
    const checkboxes = container.querySelectorAll('input[type="checkbox"]');
    expect(checkboxes.length).toBe(2);

    // Toggle transB
    const transB = checkboxes[1] as HTMLInputElement;
    transB.checked = true;
    transB.dispatchEvent(new window.Event('change', { bubbles: true }));
    expect(mutator.graph.nodes[0]!.attributes['transB']!.value).toBe(1);
  });

  it('113. Split editor', () => {
    renderNodeOp('Split', {
      split: { name: 'split', type: 'INTS', value: [2, 2] },
    });
    expect(container.textContent).toContain('Split Settings');
  });

  it('114. Resize editor', () => {
    renderNodeOp('Resize', {
      mode: { name: 'mode', type: 'STRING', value: 'linear' },
    });
    expect(container.textContent).toContain('Resize Settings');
    const select = container.querySelector('select') as HTMLSelectElement;
    expect(select.value).toBe('linear');

    select.value = 'cubic';
    select.dispatchEvent(new window.Event('change', { bubbles: true }));
    expect(mutator.graph.nodes[0]!.attributes['mode']!.value).toBe('cubic');
  });

  it('115. Squeeze editor with and without axes attr', () => {
    // With attr
    renderNodeOp('Squeeze', {
      axes: { name: 'axes', type: 'INTS', value: [1] },
    });
    expect(container.textContent).toContain('Squeeze Settings');
    expect(container.querySelector('input[value="[1]"]')).not.toBeNull();

    // Without attr (opset >= 13)
    container.innerHTML = '';
    renderNodeOp('Unsqueeze');
    expect(container.textContent).toContain('Axes are defined as an input');
  });

  it('116. Cast editor', () => {
    const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => undefined);
    renderNodeOp('Cast', {
      to: { name: 'to', type: 'INT', value: 1 }, // FLOAT
    });
    expect(container.textContent).toContain('Cast Settings');
    const select = container.querySelector('select') as HTMLSelectElement;
    expect(select.options[select.selectedIndex]!.textContent).toBe('FLOAT');

    select.value = '7'; // INT64
    select.dispatchEvent(new window.Event('change', { bubbles: true }));
    expect(mutator.graph.nodes[0]!.attributes['to']!.value).toBe(7);
  });

  it('117. Constant editor', () => {
    renderNodeOp('Constant', {
      value: { name: 'value', type: 'TENSOR', value: new Float32Array([1.0, 2.0]) },
    });
    expect(container.textContent).toContain('Constant Value');
    expect(container.textContent).toContain('1');
    expect(container.textContent).toContain('2');
  });

  it('117. Constant editor raw value fallback', () => {
    renderNodeOp('Constant', {
      value: { name: 'value', type: 'FLOAT', value: 1.5 },
    });
    expect(container.textContent).toContain('1.5');
  });

  it('118. 119. If / Loop editors', () => {
    let fired = false;
    window.addEventListener(
      'open-subgraph',
      (e) => {
        fired = true;
        expect(e.detail.name).toContain('then_branch');
      },
      { once: true },
    );
    renderNodeOp('If', {
      then_branch: { name: 'then_branch', type: 'GRAPH', value: new Graph('Then') },
    });
    expect(container.textContent).toContain('If Branch Navigation');

    const btn = container.querySelector('button') as HTMLButtonElement;
    btn.click();
    expect(fired).toBe(true);

    renderNodeOp('Loop', {
      body: { name: 'body', type: 'GRAPH', value: new Graph('Body') },
    });
    expect(container.textContent).toContain('Loop Body Navigation');
  });

  it('handles invalid JSON in array input gracefully', () => {
    renderNodeOp('Conv');
    const stridesInput = Array.from(container.querySelectorAll('input')).find(
      (i) => i.type === 'text' && i.value === '[]',
    ) as HTMLInputElement;
    stridesInput.value = '[bad json}';
    stridesInput.dispatchEvent(new window.Event('change', { bubbles: true }));
    // Should not crash, attribute should be undefined still or original
    expect(mutator.graph.nodes[0]!.attributes['strides']).toBeUndefined();
  });

  it('handles bad number input gracefully', () => {
    renderNodeOp('Gemm');
    const alphaInput = Array.from(container.querySelectorAll('input')).find(
      (i) => i.type === 'number' && i.value === '1',
    ) as HTMLInputElement;
    alphaInput.value = 'invalid';
    alphaInput.dispatchEvent(new window.Event('change', { bubbles: true }));
    expect(mutator.graph.nodes[0]!.attributes['alpha']).toBeUndefined();
  });
});
