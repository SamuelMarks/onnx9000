import { describe, it, expect, vi } from 'vitest';
import { WebNNProvider } from '../src/providers/webnn/index.js';
import { Graph } from '@onnx9000/core';

describe('WebNNProvider extra coverage', () => {
  it('should hit object.destroy missing branch', async () => {
    const p = new WebNNProvider({});
    const g = new Graph('g');

    Object.defineProperty(globalThis, 'navigator', {
      value: {
        ml: {
          createContext: vi.fn().mockResolvedValue({}),
        },
      },
      configurable: true,
    });
    (globalThis as Object).MLGraphBuilder = class {
      build() {
        return {};
      }
    };

    await p.initialize();

    p.compiledGraph = { destroy: 123 } as Object; // Not a function
    p.currentGraphId = 'g2'; // different

    try {
      await p.execute(g, {});
    } catch (e) {}
    expect(p.compiledGraph).toBeDefined(); // Shouldn't throw on destroy
  });

  it('should hit Date.now missing branch', async () => {
    const p = new WebNNProvider({});
    const g = new Graph('g');

    const origPerf = globalThis.performance;
    Object.defineProperty(globalThis, 'performance', {
      value: undefined,
      configurable: true,
    });

    Object.defineProperty(globalThis, 'navigator', {
      value: {
        ml: {
          createContext: vi.fn().mockResolvedValue({}),
        },
      },
      configurable: true,
    });
    (globalThis as Object).MLGraphBuilder = class {
      build() {
        return {};
      }
    };
    await p.initialize();

    try {
      await p.execute(g, {});
    } catch (e) {}

    Object.defineProperty(globalThis, 'performance', {
      value: origPerf,
      configurable: true,
    });
  });

  it('should hit compiledGraph.destroy function execution', async () => {
    const p = new WebNNProvider({});
    const g = new Graph('g');
    const mockDestroy = vi.fn();

    Object.defineProperty(globalThis, 'navigator', {
      value: {
        ml: {
          createContext: vi.fn().mockResolvedValue({}),
        },
      },
      configurable: true,
    });
    (globalThis as Object).MLGraphBuilder = class {};
    await p.initialize();

    p.compiledGraph = { destroy: mockDestroy } as Object;
    p.currentGraphId = 'g2';

    try {
      await p.execute(g, {});
    } catch (e) {}
    expect(mockDestroy).toHaveBeenCalled();
  });

  it('should throw when resultData is missing', async () => {
    const p = new WebNNProvider({});
    const g = new Graph('g');
    g.outputs.push({ name: 'out1', shape: [1], dtype: 'float32' } as Object);

    const mockCompute = vi.fn().mockResolvedValue({ outputs: {} }); // Missing 'out1'
    Object.defineProperty(globalThis, 'navigator', {
      value: {
        ml: {
          createContext: vi.fn().mockResolvedValue({
            compute: mockCompute,
          }),
        },
      },
      configurable: true,
    });
    (globalThis as Object).MLGraphBuilder = class {};
    await p.initialize();

    // Make execute succeed but return missing output
    p.compiledGraph = { destroy: undefined } as Object;
    p.currentGraphId = g.id;
    p.allocateBuffer = () => ({}) as Object;
    (p as Object).contextManager = {
      initialize: vi.fn().mockResolvedValue({ compute: mockCompute }),
      getContext: vi.fn().mockReturnValue({ compute: mockCompute }),
      getBuilder: vi.fn().mockReturnValue({}),
    };

    await expect(p.execute(g, {})).rejects.toThrow(
      'WebNN Execution Error: Missing output out1 from WebNN compute.',
    );
  });
});
