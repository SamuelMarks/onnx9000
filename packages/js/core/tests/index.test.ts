import { describe, it, expect } from 'vitest';
import { load, save } from '../src/index.js';
import { Graph, ValueInfo } from '../src/ir/graph.js';
import { Node } from '../src/ir/node.js';

describe('index', () => {
  it('should save and load a graph', async () => {
    const graph = new Graph('index_test');
    graph.inputs.push(new ValueInfo('x', [1], 'float32'));
    graph.outputs.push(new ValueInfo('y', [1], 'float32'));
    graph.nodes.push(new Node('Relu', ['x'], ['y'], {}, 'relu1'));

    const buffer = await save(graph);
    expect(buffer).toBeInstanceOf(ArrayBuffer);

    const loadedGraph = await load(buffer);
    expect(loadedGraph.name).toBe('index_test');
    expect(loadedGraph.nodes.length).toBe(1);
    expect(loadedGraph.nodes[0].opType).toBe('Relu');
  });

  it('should load from Uint8Array', async () => {
    const graph = new Graph('uint8_test');
    graph.inputs.push(new ValueInfo('x', [1], 'float32'));
    graph.outputs.push(new ValueInfo('y', [1], 'float32'));
    graph.nodes.push(new Node('Relu', ['x'], ['y'], {}, 'relu1'));

    const buffer = await save(graph);
    const uint8 = new Uint8Array(buffer);

    const loadedGraph = await load(uint8);
    expect(loadedGraph.name).toBe('uint8_test');
  });
});
