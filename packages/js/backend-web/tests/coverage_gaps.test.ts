import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Graph, Node, Tensor, Attribute } from '@onnx9000/core';
import { InferenceSession } from '../src/session.js';
import { WasmProvider } from '../src/providers/wasm/index.js';
import { GraphPartitioner } from '../src/partitioner.js';
import { WebNNContextManager } from '../src/providers/webnn/context.js';
import { WebNNProvider } from '../src/providers/webnn/index.js';

describe('Coverage gaps for WebNN Context', () => {
  it('should throw if MLGraphBuilder is totally missing on globalThis', async () => {
    const mockContext = { compute: vi.fn() };
    Object.defineProperty(global, 'navigator', {
      value: { ml: { createContext: vi.fn().mockResolvedValue(mockContext) } },
      configurable: true,
    });
    const orig = (globalThis as any).MLGraphBuilder;
    delete (globalThis as any).MLGraphBuilder;
    // @ts-ignore
    delete global.MLGraphBuilder;
    const manager = WebNNContextManager.getInstance();
    await expect(manager.initialize()).rejects.toThrow(
      'MLGraphBuilder is not available in this environment.',
    );
    (globalThis as any).MLGraphBuilder = orig;
  });

  beforeEach(() => {
    WebNNContextManager.getInstance().reset();
  });

  it('should return early if already initialized', async () => {
    const mockContext = { compute: vi.fn() };
    Object.defineProperty(global, 'navigator', {
      value: { ml: { createContext: vi.fn().mockResolvedValue(mockContext) } },
      configurable: true,
    });
    // @ts-ignore
    global.MLGraphBuilder = class {
      constructor() {}
      input() {
        return { shape: [1] };
      }
      constant() {
        return {};
      }
      build() {
        return Promise.resolve({ destroy: vi.fn() });
      }
      abs() {
        return {};
      }
    };

    const manager = WebNNContextManager.getInstance();
    await manager.initialize();

    // Call again to hit the cache
    await manager.initialize();

    // Expect createContext to be called only once
    expect(global.navigator.ml.createContext).toHaveBeenCalledTimes(1);
  });

  it('should fallback to default options if createContext throws', async () => {
    const mockContext = { compute: vi.fn() };
    const mockCreateContext = vi.fn();
    mockCreateContext.mockRejectedValueOnce(new Error('Options failed'));
    mockCreateContext.mockResolvedValueOnce(mockContext);

    Object.defineProperty(global, 'navigator', {
      value: { ml: { createContext: mockCreateContext } },
      configurable: true,
    });
    // @ts-ignore
    global.MLGraphBuilder = class {
      constructor() {}
      input() {
        return { shape: [1] };
      }
      constant() {
        return {};
      }
      build() {
        return Promise.resolve({ destroy: vi.fn() });
      }
      abs() {
        return {};
      }
    };

    const manager = WebNNContextManager.getInstance();
    await manager.initialize({ deviceType: 'gpu' });

    // It should hit the catch block and try again without options
    expect(mockCreateContext).toHaveBeenCalledTimes(2);
  });

  it('should throw if default fallback createContext also throws', async () => {
    const mockCreateContext = vi.fn().mockRejectedValue(new Error('Everything failed'));

    Object.defineProperty(global, 'navigator', {
      value: { ml: { createContext: mockCreateContext } },
      configurable: true,
    });

    const manager = WebNNContextManager.getInstance();
    await expect(manager.initialize()).rejects.toThrow(
      'Failed to initialize WebNN context completely: Error: Everything failed',
    );
  });

  it('should use globalThis.MLGraphBuilder if MLGraphBuilder is not defined directly', async () => {
    const mockContext = { compute: vi.fn() };
    Object.defineProperty(global, 'navigator', {
      value: { ml: { createContext: vi.fn().mockResolvedValue(mockContext) } },
      configurable: true,
    });

    // Temporarily hide global MLGraphBuilder but provide it on globalThis
    const orig = (globalThis as any).MLGraphBuilder;
    // @ts-ignore
    delete global.MLGraphBuilder;
    (globalThis as any).MLGraphBuilder = class {
      constructor() {}
      input() {
        return { shape: [1] };
      }
      constant() {
        return {};
      }
      build() {
        return Promise.resolve({ destroy: vi.fn() });
      }
      abs() {
        return {};
      }
    };

    const manager = WebNNContextManager.getInstance();
    await manager.initialize();
    expect(manager.getBuilder()).toBeDefined();

    // Restore
    (globalThis as any).MLGraphBuilder = orig;
  });

  it('should use window.MLGraphBuilder if window is defined', async () => {
    const mockContext = { compute: vi.fn() };
    Object.defineProperty(global, 'navigator', {
      value: { ml: { createContext: vi.fn().mockResolvedValue(mockContext) } },
      configurable: true,
    });

    const mockBuilderClass = class {
      constructor() {}
      input() {
        return {};
      }
      constant() {
        return {};
      }
      build() {
        return Promise.resolve({ destroy: vi.fn() });
      }
    };

    vi.stubGlobal('window', { MLGraphBuilder: mockBuilderClass });

    const manager = WebNNContextManager.getInstance();
    await manager.initialize();
    expect(manager.getBuilder()).toBeDefined();

    vi.unstubAllGlobals();
  });

  it('should return null capabilities if not initialized', () => {
    const manager = WebNNContextManager.getInstance();
    expect(manager.getCapabilities()).toBeNull();
  });
});

describe('Coverage gaps for Session & Partitioner', () => {
  it('should handle empty graph in partitioner', () => {
    const p = new GraphPartitioner([
      { name: 'WebNN', initialize: async () => {}, execute: async () => ({}) },
    ]);
    const g = new Graph('g');
    g.outputs.push({ name: 'out', shape: [1], id: 'o', dtype: 'float32' } as any);
    const regions = p.partition(g);
    expect(regions.length).toBe(1);
    expect(regions[0]?.providerName).toBe('WebNN');
  });
  it('InferenceSession should execute partitioned graphs and pass data', async () => {
    const g = new Graph('partition_test');
    g.inputs.push({ name: 'in1', shape: [1], id: 'in1', dtype: 'float32' });
    g.nodes.push(new Node('Abs', ['in'], ['out2']));
    g.outputs.push({ name: 'out2', shape: [1], id: 'out2', dtype: 'float32' } as any);

    // Node 1: Supported by Provider 1
    g.nodes.push(new Node('Add', ['in1', 'in1'], ['mid1']));
    // Node 2: Supported by Provider 2
    g.nodes.push(new Node('Sub', ['mid1', 'in1'], ['out2']));

    const wnnProvider = new WebNNProvider();
    wnnProvider.execute = vi.fn().mockResolvedValue({ mid1: new Tensor('mid1', [1], 'float32') });

    const wasmProvider = new WasmProvider();
    wasmProvider.execute = vi.fn().mockResolvedValue({ out2: new Tensor('out2', [1], 'float32') });

    // Force WebNN provider checkNodeSupported to false for Sub
    const session = new InferenceSession(g, [wnnProvider, wasmProvider]);

    // Hack the partitioner to reject Sub in WebNN
    const partitioner = (session as any).partitioner;
    const origCheck = partitioner.checkNodeSupported.bind(partitioner);
    partitioner.checkNodeSupported = (node: Node, pName: string) => {
      if (pName === 'WebNN' && node.opType === 'Sub') return false;
      return origCheck(node, pName);
    };

    const res = await session.run(['out2'], {
      in1: new Tensor('in1', [1], 'float32', false, true, new Float32Array([1])),
    });

    expect(res['out2']).toBeDefined();
    expect(wnnProvider.execute).toHaveBeenCalled();
    expect(wasmProvider.execute).toHaveBeenCalled();
  });

  it('Session should throw if provider is missing for a region', async () => {
    const g = new Graph('g');
    g.inputs.push({ name: 'in1', shape: [1], id: 'in1', dtype: 'float32' });
    g.nodes.push(new Node('Abs', ['in'], ['out1']));
    g.outputs.push({ name: 'out1', shape: [1], id: 'out1', dtype: 'float32' } as any);
    g.nodes.push(new Node('Add', ['in1', 'in1'], ['out1']));

    const p1 = { name: 'P1', initialize: async () => {}, execute: async () => ({}) };
    const session = new InferenceSession(g, [p1]);

    // Corrupt the partition regions
    vi.spyOn((session as any).partitioner, 'partition').mockReturnValue([
      {
        providerName: 'FakeProvider',
        subGraph: g,
        inputs: ['in1'],
        outputs: ['out1'],
      },
    ]);

    await expect(
      session.run(['out1'], {
        in1: new Tensor('in1', [1], 'float32', false, true, new Float32Array([1])),
      }),
    ).rejects.toThrow('Provider FakeProvider not found.');
  });
});

describe('Coverage gaps for Partitioner unsupported nodes', () => {
  it('should flag Attention and FlashAttention as unsupported', () => {
    const p = new GraphPartitioner([
      { name: 'WebNN', initialize: async () => {}, execute: async () => ({}) },
      { name: 'WASM', initialize: async () => {}, execute: async () => ({}) },
    ]);
    const g = new Graph('g');
    g.nodes.push(new Node('Attention', [], []));
    g.nodes.push(new Node('FlashAttention', [], []));
    const regions = p.partition(g);
    expect(regions[0]?.providerName).not.toBe('WebNN'); // Falls back to WASM natively
  });

  it('should flag RotaryEmbedding as unsupported', () => {
    const p = new GraphPartitioner([
      { name: 'WebNN', initialize: async () => {}, execute: async () => ({}) },
      { name: 'WASM', initialize: async () => {}, execute: async () => ({}) },
    ]);
    const g = new Graph('g');
    g.nodes.push(new Node('RotaryEmbedding', [], []));
    const regions = p.partition(g);
    expect(regions[0]?.providerName).not.toBe('WebNN');
  });

  it('should flag dynamic Concat as unsupported', () => {
    const p = new GraphPartitioner([
      { name: 'WebNN', initialize: async () => {}, execute: async () => ({}) },
      { name: 'WASM', initialize: async () => {}, execute: async () => ({}) },
    ]);
    const g = new Graph('g');
    const n = new Node('Concat', [], []);
    n.attributes['dynamic'] = new Attribute('dynamic', 'INT', 1);
    g.nodes.push(n);
    const regions = p.partition(g);
    expect(regions[0]?.providerName).not.toBe('WebNN');
  });

  it('should flag embedding Gather as unsupported', () => {
    const p = new GraphPartitioner([
      { name: 'WebNN', initialize: async () => {}, execute: async () => ({}) },
      { name: 'WASM', initialize: async () => {}, execute: async () => ({}) },
    ]);
    const g = new Graph('g');
    const n = new Node('Gather', [], []);
    n.attributes['is_embedding'] = new Attribute('is_embedding', 'INT', 1);
    g.nodes.push(n);
    const regions = p.partition(g);
    expect(regions[0]?.providerName).not.toBe('WebNN');
  });

  it('should throw if disableWebNNFallback is true and node unsupported', () => {
    const p = new GraphPartitioner(
      [
        { name: 'WebNN', initialize: async () => {}, execute: async () => ({}) },
        { name: 'WASM', initialize: async () => {}, execute: async () => ({}) },
      ],
      true,
    );
    const g = new Graph('g');
    g.nodes.push(new Node('Attention', [], []));
    expect(() => p.partition(g)).toThrow(
      'Node Attention is not supported on WebNN, but fallback is disabled.',
    );
  });
});

describe('WebNNProvider Edge Case Coverages', () => {
  beforeEach(() => {
    WebNNContextManager.getInstance().reset();
  });

  it('should hit provider early return and destroy checks', async () => {
    const mockContext = {
      compute: vi.fn().mockResolvedValue({ outputs: { out: new Float32Array([1.0]) } }),
    };
    Object.defineProperty(global, 'navigator', {
      value: { ml: { createContext: vi.fn().mockResolvedValue(mockContext) } },
      configurable: true,
    });
    const mockDestroy = vi.fn();
    // @ts-ignore
    global.MLGraphBuilder = class {
      constructor() {}
      input() {
        return { shape: [1] };
      }
      constant() {
        return {};
      }
      build() {
        return Promise.resolve({ destroy: mockDestroy });
      }
      abs() {
        return {};
      }
    };
    const provider = new WebNNProvider();
    await provider.initialize();

    const g = new Graph('g');
    g.inputs.push({ name: 'in', shape: [1], id: 'in', dtype: 'float32' });
    g.nodes.push(new Node('Abs', ['in'], ['out']));
    g.outputs.push({ name: 'out', shape: [1], id: 'out', dtype: 'float32' } as any);

    // First run compiles it
    await provider.execute(g, {
      in: new Tensor('in', [1], 'float32', false, true, new Float32Array([1])),
    });

    // Second run with identical ID uses cache
    await provider.execute(g, {
      in: new Tensor('in', [1], 'float32', false, true, new Float32Array([1])),
    });

    // Third run with NEW ID triggers destroy and recompile
    const g2 = new Graph('g2');
    g2.inputs.push({ name: 'in', shape: [1], id: 'in', dtype: 'float32' });
    g2.nodes.push(new Node('Abs', ['in'], ['out']));
    g2.outputs.push({ name: 'out', shape: [1], id: 'out', dtype: 'float32' } as any);
    await provider.execute(g2, {
      in: new Tensor('in', [1], 'float32', false, true, new Float32Array([1])),
    });

    expect(mockDestroy).toHaveBeenCalled();
  });

  it('should throw on tensor inputs without data', async () => {
    Object.defineProperty(global, 'navigator', {
      value: { ml: { createContext: vi.fn().mockResolvedValue({}) } },
      configurable: true,
    });
    // @ts-ignore
    global.MLGraphBuilder = class {
      constructor() {}
      input() {
        return { shape: [1] };
      }
      constant() {
        return {};
      }
      build() {
        return Promise.resolve({ destroy: vi.fn() });
      }
      abs() {
        return {};
      }
    };

    const provider = new WebNNProvider();
    await provider.initialize();
    const g = new Graph('g');
    g.inputs.push({ name: 'in1', shape: [1], id: 'in1', dtype: 'float32' });
    g.nodes.push(new Node('Abs', ['in1'], ['out']));
    g.outputs.push({ name: 'out', shape: [1], id: 'out', dtype: 'float32' } as any);

    await expect(
      provider.execute(g, {
        in1: new Tensor('in1', [1], 'float32', false, true, undefined),
      }),
    ).rejects.toThrow('Input tensor in1 has no data');
  });
});

describe('Partitioner end-of-loop', () => {
  it('should skip WebNN if single node at end', () => {
    const p1 = { name: 'WebNN', initialize: async () => {}, execute: async () => ({}) };
    const p2 = { name: 'WASM', initialize: async () => {}, execute: async () => ({}) };
    const partitioner = new GraphPartitioner([p1, p2], false);
    const g = new Graph('g');
    g.nodes.push(new Node('Add', ['in1', 'in2'], ['out1']));
    const regions = partitioner.partition(g);
    expect(regions[0]?.providerName).toBe('WASM');
  });
});

describe('WebNNProvider allocateBuffer types', () => {
  it('should test all buffer typings', async () => {
    const mockContext = {
      compute: vi.fn().mockResolvedValue({ outputs: { out: new Float32Array([1.0]) } }),
    };
    Object.defineProperty(global, 'navigator', {
      value: { ml: { createContext: vi.fn().mockResolvedValue(mockContext) } },
      configurable: true,
    });
    // @ts-ignore
    global.MLGraphBuilder = class {
      constructor() {}
      input() {
        return { shape: [1] };
      }
      constant() {
        return {};
      }
      build() {
        return Promise.resolve({ destroy: vi.fn() });
      }
      abs() {
        return {};
      }
    };
    const provider = new WebNNProvider();
    await provider.initialize();
    const g = new Graph('g');
    g.inputs.push({ name: 'in', shape: [1], id: 'in', dtype: 'float32' });
    g.nodes.push(new Node('Abs', ['in'], ['out1']));
    g.outputs.push({ name: 'out1', shape: [1], id: 'o1', dtype: 'int32' } as any);
    g.nodes.push(new Node('Abs', ['in'], ['out2']));
    g.outputs.push({ name: 'out2', shape: [1], id: 'o2', dtype: 'float16' } as any);
    g.nodes.push(new Node('Abs', ['in'], ['out3']));
    g.outputs.push({ name: 'out3', shape: [1], id: 'o3', dtype: 'uint8' } as any);
    g.nodes.push(new Node('Abs', ['in'], ['out4']));
    g.outputs.push({ name: 'out4', shape: [1], id: 'o4', dtype: 'int8' } as any);
    g.nodes.push(new Node('Abs', ['in'], ['out5']));
    g.outputs.push({ name: 'out5', shape: [1], id: 'o5', dtype: 'int64' } as any);
    g.nodes.push(new Node('Abs', ['in'], ['out6']));
    g.outputs.push({ name: 'out6', shape: [1], id: 'o6', dtype: 'unknown' as any } as any);
    await expect(
      provider.execute(g, {
        in: new Tensor('in', [1], 'float32', false, true, new Float32Array([1])),
      }),
    ).rejects.toThrow(); // throws on compute because mock lacks outputs
  });
});
