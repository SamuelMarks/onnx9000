import { describe, it, expect, beforeEach } from 'vitest';
import {
  PolyfillML,
  PolyfillMLContext,
  PolyfillMLGraphBuilder,
  PolyfillMLGraph,
  PolyfillMLOperand,
} from '../src/index.js';

describe('WebNN Polyfill Environment & Context', () => {
  it('should create context', async () => {
    const ml = new PolyfillML();
    const context = await ml.createContext();
    expect(context).toBeInstanceOf(PolyfillMLContext);
    const limits = context.opSupportLimits();
    expect(limits.input.dataTypes).toContain('float32');
  });

  it('should initialize MLGraphBuilder', async () => {
    const ml = new PolyfillML();
    const context = await ml.createContext({ deviceType: 'cpu' });
    const builder = new PolyfillMLGraphBuilder(context);
    expect(builder).toBeDefined();
  });
});

describe('WebNN Polyfill Core Operators', () => {
  let context: PolyfillMLContext;
  let builder: PolyfillMLGraphBuilder;

  beforeEach(async () => {
    context = await new PolyfillML().createContext({ deviceType: 'cpu' });
    builder = new PolyfillMLGraphBuilder(context);
  });

  it('should build and execute Add operation natively', async () => {
    const inputA = builder.input('A', { dataType: 'float32', dimensions: [1, 4] });
    const inputB = builder.input('B', { dataType: 'float32', dimensions: [1, 4] });
    const output = builder.add(inputA, inputB);

    const graph = await builder.build({ C: output });
    expect(graph).toBeInstanceOf(PolyfillMLGraph);

    // Check AST construction
    const polyGraph = graph as PolyfillMLGraph;
    const addNode = polyGraph.onnxGraph.nodes.find((n) => n.opType === 'Add');
    expect(addNode).toBeDefined();
    expect(addNode!.inputs).toContain('A');
    expect(addNode!.inputs).toContain('B');

    const bufferA = new Float32Array([1, 2, 3, 4]);
    const bufferB = new Float32Array([10, 20, 30, 40]);
    const bufferC = new Float32Array(4);

    const result = await context.compute(polyGraph, { A: bufferA, B: bufferB }, { C: bufferC });

    expect(result.inputs).toBeDefined();
    expect(result.outputs).toBeDefined();
  });

  it('should handle Matrix Multiplication', async () => {
    const inputA = builder.input('A', { dataType: 'float32', dimensions: [2, 2] });
    const inputB = builder.input('B', { dataType: 'float32', dimensions: [2, 2] });
    const output = builder.matmul(inputA, inputB);

    const graph = await builder.build({ C: output });

    // Check AST
    const polyGraph = graph as PolyfillMLGraph;
    const matMulNode = polyGraph.onnxGraph.nodes.find((n) => n.opType === 'MatMul');
    expect(matMulNode).toBeDefined();

    const bufferA = new Float32Array([1, 2, 3, 4]);
    const bufferB = new Float32Array([1, 0, 0, 1]); // Identity matrix
    const bufferC = new Float32Array(4);

    await context.compute(polyGraph, { A: bufferA, B: bufferB }, { C: bufferC });

    expect(polyGraph.onnxGraph.nodes.length).toBeGreaterThan(0);
  });

  it('should run MLTensor dispatch mapping natively', async () => {
    const inputA = builder.input('A', { dataType: 'float32', dimensions: [1] });
    const output = builder.relu(inputA);
    const graph = await builder.build({ C: output });

    const tensorA = await context.createTensor({ dataType: 'float32', dimensions: [1] });
    const tensorC = await context.createTensor({ dataType: 'float32', dimensions: [1] });

    // Write data natively
    await context.writeTensor(tensorA, new Float32Array([-5]).buffer);

    // Dispatch
    await context.dispatch(graph as PolyfillMLGraph, { A: tensorA }, { C: tensorC });

    // Read data natively
    const outBuf = new Float32Array(1);
    await context.readTensor(tensorC, outBuf.buffer);

    expect(outBuf[0]).toBe(0);

    tensorA.destroy();
    tensorC.destroy();
  });
});
