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

describe('Tensor Format', () => {
  it('should format no data', () => {
    const t = new Tensor('T', [2], 'float32');
    expect(t.formatData()).toBe('No data');
  });

  it('should generate uuid via math random if crypto missing', () => {
    // we can mock crypto to test the fallback
    const origCrypto = globalThis.crypto;
    // @ts-ignore
    delete globalThis.crypto;
    const t = new Tensor('T', [2], 'float32');
    expect(t.id).toBeDefined();
    expect(typeof t.id).toBe('string');
    // @ts-ignore
    globalThis.crypto = origCrypto;
  });

  it('should store external data', () => {
    const ext = { location: 'model.data', offset: 0, length: 100 };
    const t = new Tensor('T', [2], 'float32', false, true, null, ext);
    expect(t.externalData).toBe(ext);
  });

  it('should format float32 data', () => {
    const buf = new Float32Array([1.0, 2.5, 3.0, 4.0]);
    const t = new Tensor('T', [4], 'float32', false, true, new Uint8Array(buf.buffer));
    const str = t.formatData(2);
    expect(str).toBe('[1, 2.5 ... +2 elements]');
  });

  it('should format int64 data', () => {
    const buf = new BigInt64Array([1n, -500n]);
    const t = new Tensor('T', [2], 'int64', false, true, new Uint8Array(buf.buffer));
    const str = t.formatData();
    expect(str).toBe('[1, -500]');
  });

  it('should format float16 data', () => {
    const buf = new Uint16Array([0x3c00, 0xbc00]); // 1.0, -1.0
    const t = new Tensor('T', [2], 'float16', false, true, new Uint8Array(buf.buffer));
    const str = t.formatData();
    expect(str).toBe('[1, -1]');
  });

  it('should format float16 zero data', () => {
    const buf = new Uint16Array([0x0000]); // 0
    const t = new Tensor('T', [1], 'float16', false, true, new Uint8Array(buf.buffer));
    const str = t.formatData();
    expect(str).toBe('[0]');
  });

  it('should format bfloat16 data', () => {
    const buf = new Uint16Array([0x3f80, 0xbf80]); // 1.0, -1.0
    const t = new Tensor('T', [2], 'bfloat16', false, true, new Uint8Array(buf.buffer));
    const str = t.formatData();
    expect(str).toBe('[1, -1]');
  });

  it('should format int8/uint8 fallback data', () => {
    const buf = new Uint8Array([10, 20]);
    const t = new Tensor('T', [2], 'uint8', false, true, buf);
    const str = t.formatData();
    expect(str).toBe('[10, 20]');
  });

  it('should format directly from typed array', () => {
    const buf = new Float32Array([10.5, 20.5]);
    const t = new Tensor('T', [2], 'float32', false, true, buf);
    const str = t.formatData();
    expect(str).toBe('[10.5, 20.5]');
  });
});

describe('Graph specific', () => {
  it('should generate vi uuid via math random if crypto missing', () => {
    const origCrypto = globalThis.crypto;
    // @ts-ignore
    delete globalThis.crypto;
    const g = new Graph('g1');
    const vi = new ValueInfo('V', [1], 'float32');
    expect(vi.id).toBeDefined();
    expect(g.id).toBeDefined();
    // @ts-ignore
    globalThis.crypto = origCrypto;
  });
});

describe('Node format/missing', () => {
  it('should handle optional properties', () => {
    const n = new Node('Relu', [], [], {}, undefined, undefined);
    expect(n.name).toBe('');
    expect(n.domain).toBe('');
  });
});

describe('Node format/missing', () => {
  it('should generate uuid via math random if crypto missing', () => {
    const origCrypto = globalThis.crypto;
    // @ts-ignore
    delete globalThis.crypto;
    const n = new Node('Relu', [], []);
    expect(n.id).toBeDefined();
    // @ts-ignore
    globalThis.crypto = origCrypto;
  });
});
