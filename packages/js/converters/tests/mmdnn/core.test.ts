import { describe, it, expect } from 'vitest';
import { convert } from '../../src/mmdnn/api.js';
import { MMDNNReporter } from '../../src/mmdnn/reporter.js';
import { topologicalSort } from '../../src/mmdnn/topology.js';
import { DataLayoutTracker } from '../../src/mmdnn/layout.js';
import { ShapeInferenceEngine } from '../../src/mmdnn/shape-inference.js';
import { NodeFusionRegistry } from '../../src/mmdnn/fusion.js';
import { FileLoader } from '../../src/mmdnn/file-loader.js';
import { Graph, ValueInfo, Node, Shape } from '@onnx9000/core';

describe('MMDNN - Core Architecture', () => {
  it('should initialize FileLoader with Blobs/Files', async () => {
    const file = new File(['dummy content'], 'model.onnx', { type: 'application/octet-stream' });
    const loader = new FileLoader([file]);
    expect(loader.hasFile('model.onnx')).toBe(true);

    const text = await loader.readText('model.onnx');
    expect(text).toBe('dummy content');
  });

  it('should topologically sort an ONNX graph', () => {
    const graph = new Graph('test_graph');

    // Create an out-of-order graph
    // nodeB consumes nodeA's output
    // nodeC consumes nodeB's output

    const nodeC = new Node('Add', ['B_out', 'input1'], ['C_out'], {}, 'NodeC');
    const nodeA = new Node('Relu', ['input0'], ['A_out'], {}, 'NodeA');
    const nodeB = new Node('Relu', ['A_out'], ['B_out'], {}, 'NodeB');

    graph.inputs = [
      new ValueInfo('input0', [1, 10], 'float32'),
      new ValueInfo('input1', [1, 10], 'float32'),
    ];
    graph.nodes = [nodeC, nodeA, nodeB];

    const reporter = new MMDNNReporter();
    const sorted = topologicalSort(graph, reporter);

    expect(sorted.nodes[0].name).toBe('NodeA');
    expect(sorted.nodes[1].name).toBe('NodeB');
    expect(sorted.nodes[2].name).toBe('NodeC');
  });

  it('should track data layout NCHW to NHWC across Transpose', () => {
    const graph = new Graph('test_layout');
    graph.inputs = [new ValueInfo('image_in', [1, 3, 224, 224], 'float32')];

    const transpose = new Node('Transpose', ['image_in'], ['image_nhwc'], {
      perm: { name: 'perm', type: 'INTS', value: [0, 2, 3, 1] },
    });
    const relu = new Node('Relu', ['image_nhwc'], ['image_out']);

    graph.nodes = [transpose, relu];

    const tracker = new DataLayoutTracker();
    const reporter = new MMDNNReporter();
    tracker.track(graph, reporter);

    expect(tracker.getLayout('image_in')).toBe('NCHW');
    expect(tracker.getLayout('image_nhwc')).toBe('NHWC');
    expect(tracker.getLayout('image_out')).toBe('NHWC');
  });

  it('should apply shape inference to broadcasting operations', () => {
    const graph = new Graph('test_shape');
    graph.inputs = [
      new ValueInfo('A', [1, 3, 224, 224], 'float32'),
      new ValueInfo('B', [3, 1, 1], 'float32'),
    ];

    const addNode = new Node('Add', ['A', 'B'], ['C']);
    graph.nodes = [addNode];

    const engine = new ShapeInferenceEngine();
    const reporter = new MMDNNReporter();
    engine.inferShapes(graph, reporter);

    const deducedShape = engine.getShape('C');
    expect(deducedShape).toBeDefined();
    expect(deducedShape!).toEqual([1, 3, 224, 224]);
  });

  it('should fuse Node sequences (MatMul + Add -> Gemm)', () => {
    const graph = new Graph('test_fusion');

    const matmul = new Node('MatMul', ['X', 'W'], ['Y']);
    const add = new Node('Add', ['Y', 'B'], ['Z']);

    graph.nodes = [matmul, add];

    const registry = new NodeFusionRegistry();
    const reporter = new MMDNNReporter();
    const fusedGraph = registry.applyFusions(graph, reporter);

    expect(fusedGraph.nodes.length).toBe(1);
    expect(fusedGraph.nodes[0].opType).toBe('Gemm');
    expect(fusedGraph.nodes[0].inputs).toEqual(['X', 'W', 'B']);
    expect(fusedGraph.nodes[0].outputs).toEqual(['Z']);
  });

  it('should execute convert API', async () => {
    const file = new File([''], 'model.onnx', { type: 'application/octet-stream' });
    const result = await convert('onnx', 'pytorch_code', [file]);
    expect(result).toBe('Exported pytorch_code content for placeholder');
  });
});
