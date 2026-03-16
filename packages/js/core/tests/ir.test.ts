import { describe, it, expect } from 'vitest';
import { Graph, ValueInfo } from '../src/ir/graph.js';
import { Node, Attribute } from '../src/ir/node.js';
import { Tensor } from '../src/ir/tensor.js';

describe('Tensor', () => {
  it('should initialize correctly', () => {
    const t = new Tensor('T1', [1, 2, 'N', -1], 'float32');
    expect(t.name).toBe('T1');
    expect(t.shape).toEqual([1, 2, 'N', -1]);
    expect(t.dtype).toBe('float32');
    expect(t.size).toBe(2); // 1 * 2, 'N' and -1 are ignored in size calc
  });
});

describe('Node and Attribute', () => {
  it('should initialize correctly', () => {
    const attr = new Attribute('a', 'FLOAT', 1.0);
    expect(attr.name).toBe('a');
    expect(attr.type).toBe('FLOAT');
    expect(attr.value).toBe(1.0);

    const node = new Node('Add', ['X'], ['Y'], { a: attr }, 'N1', 'ai.onnx');
    expect(node.opType).toBe('Add');
    expect(node.inputs).toEqual(['X']);
    expect(node.outputs).toEqual(['Y']);
    expect(node.name).toBe('N1');
    expect(node.domain).toBe('ai.onnx');
    expect(node.attributes['a'].value).toBe(1.0);

    // Default values
    const n2 = new Node('Sub', [], []);
    expect(n2.attributes).toEqual({});
    expect(n2.name).toBe('');
    expect(n2.domain).toBe('');
  });
});

describe('Graph and ValueInfo', () => {
  it('should manage components correctly', () => {
    const vi = new ValueInfo('V', [1], 'float32');
    expect(vi.name).toBe('V');
    expect(vi.shape).toEqual([1]);
    expect(vi.dtype).toBe('float32');

    const g = new Graph('my_graph');
    expect(g.name).toBe('my_graph');

    const t = new Tensor('T', [1], 'float32');
    g.addTensor(t);
    expect(g.tensors['T']).toBe(t);

    const n = new Node('Relu', [], [], {}, 'Node1');
    g.addNode(n);
    expect(g.nodes).toContain(n);

    expect(g.getNode('Node1')).toBe(n);
    expect(g.getNode('Missing')).toBeNull();
  });
});

describe('Index Export', () => {
  it('should export all components', async () => {
    const mod = await import('../src/index.js');
    expect(mod.Tensor).toBeDefined();
    expect(mod.Node).toBeDefined();
    expect(mod.Graph).toBeDefined();
    expect(mod.ValueInfo).toBeDefined();
    expect(mod.Attribute).toBeDefined();
  });
});
