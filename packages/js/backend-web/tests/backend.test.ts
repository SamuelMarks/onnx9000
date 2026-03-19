import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Graph, Tensor, Node } from '@onnx9000/core';
import { InferenceSession } from '../src/session.js';
import { WebGPUProvider } from '../src/providers/webgpu/index.js';
import { WasmProvider } from '../src/providers/wasm/index.js';
import { WebNNProvider } from '../src/providers/webnn/index.js';
import { WebNNContextManager } from '../src/providers/webnn/context.js';
import * as index from '../src/index.js';

describe('Index Export', () => {
  it('should export all components', () => {
    expect(index.InferenceSession).toBeDefined();
    expect(index.WebGPUProvider).toBeDefined();
    expect(index.WasmProvider).toBeDefined();
    expect(index.WebNNProvider).toBeDefined();
  });
});

describe('InferenceSession', () => {
  it('should run successfully with a provider', async () => {
    const g = new Graph('g');
    g.outputs.push('out' as any);

    const provider = new WasmProvider();
    await provider.initialize();

    const session = new InferenceSession(g, [provider]);
    const res = await session.run(['out'], {});

    expect(res['out']).toBeDefined();
    expect(res['out'].name).toBe('out');
  });

  it('should throw error if no providers', async () => {
    const g = new Graph('g');
    const session = new InferenceSession(g, []);

    await expect(session.run(['out'], {})).rejects.toThrow('No Execution Providers registered.');
  });
});

describe('WebGPUProvider', () => {
  it('should throw error if navigator.gpu is missing', async () => {
    const provider = new WebGPUProvider();
    // In Node.js environment, navigator is undefined unless mocked
    await expect(provider.initialize()).rejects.toThrow('WebGPU is not supported');
  });

  it('should execute correctly', async () => {
    const provider = new WebGPUProvider();
    const g = new Graph('g');
    g.outputs.push('out' as any);
    const res = await provider.execute(g, {});
    expect(res['out']).toBeDefined();
  });
});

describe('WebNNProvider', () => {
  beforeEach(() => {
    WebNNContextManager.getInstance().reset();
  });

  it('should throw error if navigator.ml is missing', async () => {
    Object.defineProperty(global, 'navigator', {
      value: undefined,
      writable: true,
      configurable: true,
    });
    const provider = new WebNNProvider();
    await expect(provider.initialize()).rejects.toThrow('WebNN is not supported');
  });

  it('should execute correctly', async () => {
    // Mock navigator.ml and MLGraphBuilder
    const mockCompute = vi.fn().mockResolvedValue({
      outputs: { out: new Float32Array([1.0]) },
    });
    const mockContext = { compute: mockCompute };

    Object.defineProperty(global, 'navigator', {
      value: { ml: { createContext: vi.fn().mockResolvedValue(mockContext) } },
      writable: true,
      configurable: true,
    });

    // @ts-ignore
    global.MLGraphBuilder = class {
      constructor() {}
      input() {
        return {};
      }
      constant() {
        return {};
      }
      build() {
        return Promise.resolve({});
      }
      add() {
        return {};
      }
      abs() {
        return {};
      }
    };

    const provider = new WebNNProvider();
    await provider.initialize();

    const g = new Graph('g');
    g.inputs.push({ name: 'in', shape: [1], id: 'in', dtype: 'float32' });
    g.outputs.push({ name: 'out', shape: [1], id: 'out', dtype: 'float32' } as any);
    g.nodes.push(new Node('Abs', ['in'], ['out']));

    const res = await provider.execute(g, {
      in: new Tensor('in', [1], 'float32', false, true, new Float32Array([1])),
    });
    expect(res['out']).toBeDefined();
    expect(res['out']?.shape).toEqual([1]);
  });
});

describe('WasmProvider', () => {
  it('should initialize and execute correctly', async () => {
    const provider = new WasmProvider();
    await provider.initialize();

    const g = new Graph('g');
    g.outputs.push('out' as any);
    const res = await provider.execute(g, {});
    expect(res['out']).toBeDefined();
  });
});

describe('WebGPUProvider coverage gap', () => {
  it('should initialize when navigator.gpu exists', async () => {
    // Mock navigator.gpu
    Object.defineProperty(global, 'navigator', {
      value: {
        gpu: {
          requestAdapter: vi.fn().mockResolvedValue({}),
        },
      },
      writable: true,
      configurable: true,
    });

    const provider = new WebGPUProvider();
    await provider.initialize();
    expect(navigator.gpu.requestAdapter).toHaveBeenCalled();
  });
});

describe('InferenceSession ORT Parity', () => {
  it('should support create from string', async () => {
    const session = await InferenceSession.create('model_url');
    expect(session).toBeDefined();
    expect(session.options).toEqual({});
  });

  it('should support create from buffer', async () => {
    const buf = new ArrayBuffer(10);
    const session = await InferenceSession.create(buf);
    expect(session).toBeDefined();
  });

  it('should support profiling flags', () => {
    const g = new Graph('g');
    const s = new InferenceSession(g, []);
    expect(s.profilingEnabled).toBe(false);
    s.startProfiling();
    expect(s.profilingEnabled).toBe(true);
    s.endProfiling();
    expect(s.profilingEnabled).toBe(false);
  });

  it('should throw on invalid input', async () => {
    const g = new Graph('g');
    const p = new WasmProvider();
    const s = new InferenceSession(g, [p]);
    // @ts-ignore
    await expect(s.run(['out'], { missing: null })).rejects.toThrow(
      'Input missing is null or undefined',
    );
  });
});

describe('WebNN Fallbacks', () => {
  beforeEach(() => {
    WebNNContextManager.getInstance().reset();
  });

  it('should throw immediately if navigator undefined', async () => {
    Object.defineProperty(global, 'navigator', {
      value: undefined,
      writable: true,
      configurable: true,
    });
    const provider = new WebNNProvider();
    await expect(provider.initialize()).rejects.toThrow('WebNN is not supported');
  });
});

describe('Provider Object Name fallback coverage', () => {
  beforeEach(() => {
    WebNNContextManager.getInstance().reset();
  });

  it('should handle Object.name in WasmProvider', async () => {
    const provider = new WasmProvider();
    const g = new Graph('g');
    g.outputs.push({ name: 'out_obj' } as any);
    const res = await provider.execute(g, {});
    expect(res['out_obj']).toBeDefined();
  });

  it('should handle Object.name in WebGPUProvider', async () => {
    const provider = new WebGPUProvider();
    const g = new Graph('g');
    g.outputs.push({ name: 'out_obj' } as any);
    const res = await provider.execute(g, {});
    expect(res['out_obj']).toBeDefined();
  });

  it('should handle Object.name in WebNNProvider', async () => {
    // Mock navigator.ml and MLGraphBuilder
    const mockCompute = vi.fn().mockResolvedValue({
      outputs: { out_obj: new Float32Array([1.0]) },
    });
    const mockContext = { compute: mockCompute };

    Object.defineProperty(global, 'navigator', {
      value: { ml: { createContext: vi.fn().mockResolvedValue(mockContext) } },
      writable: true,
      configurable: true,
    });

    // @ts-ignore
    global.MLGraphBuilder = class {
      constructor() {}
      input() {
        return {};
      }
      constant() {
        return {};
      }
      build() {
        return Promise.resolve({});
      }
      add() {
        return {};
      }
      abs() {
        return {};
      }
    };

    const provider = new WebNNProvider();
    await provider.initialize();

    const g = new Graph('g');
    g.inputs.push({ name: 'in', shape: [1], id: 'in', dtype: 'float32' });
    g.outputs.push({ name: 'out_obj', shape: [1], dtype: 'float32' } as any);
    g.nodes.push(new Node('Abs', ['in'], ['out_obj']));

    const res = await provider.execute(g, {
      in: new Tensor('in', [1], 'float32', false, true, new Float32Array([1])),
    });
    expect(res['out_obj']).toBeDefined();
  });
});

import { GraphPartitioner } from '../src/partitioner.js';

describe('GraphPartitioner', () => {
  it('should generate distinct sub-graphs for fallback regions', () => {
    const p1 = { name: 'WebNN', initialize: async () => {}, execute: async () => ({}) };
    const p2 = { name: 'WASM', initialize: async () => {}, execute: async () => ({}) };

    const partitioner = new GraphPartitioner([p1, p2], false);

    const g = new Graph('g');
    g.nodes.push(new Node('Add', ['in1', 'in2'], ['out1']));
    g.nodes.push(new Node('NonZero', ['out1'], ['out2']));
    g.nodes.push(new Node('Mul', ['out2', 'in3'], ['out3']));

    const regions = partitioner.partition(g);
    expect(regions.length).toBeGreaterThan(1);
    expect(regions[1]!.providerName).toBe('WASM');
  });

  it('should respect disableWebNNFallback flag', () => {
    const p1 = { name: 'WebNN', initialize: async () => {}, execute: async () => ({}) };
    const partitioner = new GraphPartitioner([p1], true);

    const g = new Graph('g');
    g.nodes.push(new Node('NonZero', ['in1'], ['out1']));

    expect(() => partitioner.partition(g)).toThrow(
      'Node NonZero is not supported on WebNN, but fallback is disabled.',
    );
  });
});
