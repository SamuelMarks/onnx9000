import { describe, it, expect, vi } from 'vitest';
import { Graph, Tensor } from '@onnx9000/core';
import { InferenceSession } from '../src/session.js';
import { WebGPUProvider } from '../src/providers/webgpu/index.js';
import { WasmProvider } from '../src/providers/wasm/index.js';
import { WebNNProvider } from '../src/providers/webnn/index.js';
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
    g.outputs.push('out');

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
    // In Node.js environment, navigator is undefined
    await expect(provider.initialize()).rejects.toThrow('WebGPU is not supported');
  });

  it('should execute correctly', async () => {
    const provider = new WebGPUProvider();
    const g = new Graph('g');
    g.outputs.push('out');
    const res = await provider.execute(g, {});
    expect(res['out']).toBeDefined();
  });
});

describe('WebNNProvider', () => {
  it('should throw error if navigator.ml is missing', async () => {
    const provider = new WebNNProvider();
    await expect(provider.initialize()).rejects.toThrow('WebNN is not supported');
  });

  it('should execute correctly', async () => {
    const provider = new WebNNProvider();
    const g = new Graph('g');
    g.outputs.push('out');
    const res = await provider.execute(g, {});
    expect(res['out']).toBeDefined();
  });
});

describe('WasmProvider', () => {
  it('should initialize and execute correctly', async () => {
    const provider = new WasmProvider();
    await provider.initialize();

    const g = new Graph('g');
    g.outputs.push('out');
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
  it('should throw immediately if navigator undefined', async () => {
    const origNav = global.navigator;
    // @ts-ignore
    global.navigator = undefined;
    const provider = new WebNNProvider();
    await expect(provider.initialize()).rejects.toThrow('WebNN is not supported');
    global.navigator = origNav;
  });
});
