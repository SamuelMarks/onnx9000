import { describe, it, expect } from 'vitest';
import { OnnxAdapter, VizGraph } from '../../src/core/OnnxAdapter';

describe('OnnxAdapter', () => {
  it('should convert a simple ONNX graph to Cytoscape elements', () => {
    const graph: VizGraph = {
      inputs: [{ name: 'input1', type: 'tensor(float)' }],
      outputs: [{ name: 'output1', type: 'tensor(float)' }],
      nodes: [
        {
          id: 'node1',
          name: 'Relu_1',
          opType: 'Relu',
          inputs: ['input1'],
          outputs: ['output1']
        }
      ]
    };

    const elements = OnnxAdapter.toCytoscape(graph);

    // Total elements: 1 input node, 1 op node, 1 edge (input->op), 1 output node, 1 edge (op->output) = 5
    expect(elements.length).toBe(5);

    const inputNode = elements.find((e) => e.group === 'nodes' && e.data.id === 'input1');
    expect(inputNode).toBeDefined();

    const opNode = elements.find((e) => e.group === 'nodes' && e.data.id === 'node1');
    expect(opNode).toBeDefined();

    const outputNode = elements.find((e) => e.group === 'nodes' && e.data.id === 'output-output1');
    expect(outputNode).toBeDefined();

    const edgeInputToOp = elements.find(
      (e) => e.group === 'edges' && e.data.source === 'input1' && e.data.target === 'node1'
    );
    expect(edgeInputToOp).toBeDefined();

    const edgeOpToOutput = elements.find(
      (e) => e.group === 'edges' && e.data.source === 'node1' && e.data.target === 'output-output1'
    );
    expect(edgeOpToOutput).toBeDefined();
  });

  it('should generate initializer nodes for unknown inputs', () => {
    const graph: VizGraph = {
      inputs: [{ name: 'input1', type: 'tensor(float)' }],
      outputs: [{ name: 'output1', type: 'tensor(float)' }],
      nodes: [
        {
          id: 'node1',
          name: 'Add_1',
          opType: 'Add',
          inputs: ['input1', 'weight_tensor'], // 'weight_tensor' is not in inputs
          outputs: ['output1']
        }
      ]
    };

    const elements = OnnxAdapter.toCytoscape(graph);

    const initNode = elements.find(
      (e) => e.group === 'nodes' && e.data.id === 'init-weight_tensor'
    );
    expect(initNode).toBeDefined();
    expect(initNode?.classes).toBe('onnx-initializer');

    const edgeInitToOp = elements.find(
      (e) =>
        e.group === 'edges' && e.data.source === 'init-weight_tensor' && e.data.target === 'node1'
    );
    expect(edgeInitToOp).toBeDefined();
  });

  it('should ignore outputs that have no source', () => {
    const graph: VizGraph = {
      inputs: [],
      outputs: [{ name: 'dangling_output', type: 'tensor(float)' }],
      nodes: []
    };

    const elements = OnnxAdapter.toCytoscape(graph);
    // Should be empty since dangling_output source is not found
    expect(elements.length).toBe(0);
  });
});
