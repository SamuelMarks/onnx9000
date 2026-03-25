import { describe, it, expect } from 'vitest';
import { compileOnnxToC, initCompiler } from '../src/index.js';
import { Graph, Node } from '@onnx9000/core';
import { CGenerator } from '../src/generator.js';

describe('@onnx9000/c-compiler', () => {
  it('should initialize compiler mock', async () => {
    const pyodide = await initCompiler();
    expect(pyodide).toBeDefined();
    expect(pyodide.initialized).toBe(true);
  });

  it('should return compiled strings for C with empty buffer', async () => {
    const result = await compileOnnxToC(new Uint8Array([0, 1, 2]), {
      prefix: 'test_',
      emitCpp: false,
    });
    expect(result.header).toContain('test_run');
    expect(result.source).toContain('test_run');
    expect(result.summary).toContain('Memory Summary');
  });

  it('should return compiled strings for C++ with empty buffer', async () => {
    const result = await compileOnnxToC(new Uint8Array([0, 1, 2]), {
      prefix: 'model_',
      emitCpp: true,
    });
    expect(result.header).toContain('namespace model_');
    expect(result.source).toContain('namespace model_');
    expect(result.summary).toContain('Memory Summary');
  });

  it('should return defaults with empty options', async () => {
    const result = await compileOnnxToC(new Uint8Array([0, 1, 2]));
    expect(result.header).toContain('model_run'); // default emitCpp: false, prefix: model_
  });
});

describe('CGenerator', () => {
  const createMockGraph = () => {
    const graph = new Graph('TestGraph');
    graph.outputs = [{ name: 'output', type: 'tensor', shape: [256] }];
    graph.inputs = [{ name: 'input_1', type: 'tensor', shape: [256] }];
    graph.initializers = ['weight_1'];

    const convNode = new Node('Conv', ['input_1', 'weight_1'], ['conv_out'], {}, 'conv1');
    const reluNode = new Node('Relu', ['conv_out'], ['relu_out'], {}, 'relu1');
    const poolNode = new Node('MaxPool', ['relu_out'], ['pool_out'], {}, 'pool1');
    const flattenNode = new Node('Flatten', ['pool_out'], ['flat_out'], {}, 'flatten1');
    const denseNode = new Node('Gemm', ['flat_out', 'weight_1'], ['dense_out'], {}, 'dense1');
    const softmaxNode = new Node('Softmax', ['dense_out'], ['output'], {}, 'softmax1');

    // Add GAP and Add just to cover those paths
    const gapNode = new Node('GlobalAveragePool', ['conv_out'], ['gap_out'], {}, 'gap1');
    const addNode = new Node('Add', ['gap_out', 'conv_out'], ['add_out'], {}, 'add1');
    const unknownNode = new Node('UnknownOp', ['add_out'], ['unknown_out'], {}, 'unknown1');

    graph.nodes.push(
      convNode,
      reluNode,
      poolNode,
      flattenNode,
      denseNode,
      softmaxNode,
      gapNode,
      addNode,
      unknownNode,
    );
    return graph;
  };

  it('generates C code properly', () => {
    const graph = createMockGraph();
    const generator = new CGenerator(graph, 'my_model_', false);

    expect(generator.generateHeader()).toContain('void my_model_run');

    const source = generator.generateSource();
    expect(source).toContain('void my_model_run');
    expect(source).toContain('// Conv -> conv_out');
    expect(source).toContain('// Relu -> relu_out');
    expect(source).toContain('// MaxPool -> pool_out');
    expect(source).toContain('// Flatten -> flat_out');
    expect(source).toContain('// Gemm -> dense_out');
    expect(source).toContain('// Softmax -> output');
    expect(source).toContain('// GlobalAveragePool -> gap_out');
    expect(source).toContain('// Add -> add_out');
    expect(source).toContain('// UnknownOp -> unknown_out');
  });

  it('generates C++ code properly', () => {
    const graph = createMockGraph();
    const generator = new CGenerator(graph, 'my_model_', true);

    expect(generator.generateHeader()).toContain('namespace my_model_');

    const source = generator.generateSource();
    expect(source).toContain('namespace my_model_');
    expect(source).toContain('std::vector<float> conv_out');
    expect(source).toContain('std::vector<float> weight_1');
  });

  it('handles empty names correctly', () => {
    const graph = new Graph('EdgeGraph');
    const node = new Node('Relu', ['0_input', ''], [''], {}, '');
    graph.nodes.push(node);

    const generator = new CGenerator(graph, 'edge_');
    const source = generator.generateSource();
    expect(source).toContain('v_0_input');
    expect(source).toContain('unnamed');
  });
});
