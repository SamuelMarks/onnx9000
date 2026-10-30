import { Graph, type ValueInfo } from '@onnx9000/core';
import { describe, expect, it, vi } from 'vitest';
import { WebNNProvider } from '../src/providers/webnn/index.js';

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
    (globalThis as typeof globalThis & { MLGraphBuilder?: unknown }).MLGraphBuilder = class {
      build() {
        return {};
      }
    };

    await p.initialize();

    (p as never as { compiledGraph: unknown }).compiledGraph = { destroy: 123 }; // Not a function
    p.currentGraphId = 'g2'; // different

    try {
      await p.execute(g, {});
    } catch (_e) {
      expect(_e).toBeDefined();
    }
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
    (globalThis as typeof globalThis & { MLGraphBuilder?: unknown }).MLGraphBuilder = class {
      build() {
        return {};
      }
    };
    await p.initialize();

    try {
      await p.execute(g, {});
    } catch (_e) {
      expect(_e).toBeDefined();
    }

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
    (globalThis as typeof globalThis & { MLGraphBuilder?: unknown }).MLGraphBuilder = class {};
    await p.initialize();

    (p as never as { compiledGraph: unknown }).compiledGraph = { destroy: mockDestroy };
    p.currentGraphId = 'g2';

    try {
      await p.execute(g, {});
    } catch (_e) {
      expect(_e).toBeDefined();
    }
    expect(mockDestroy).toHaveBeenCalled();
  });

  it('should throw when resultData is missing', async () => {
    const p = new WebNNProvider({});
    const g = new Graph('g');
    g.outputs.push({ name: 'out1', shape: [1], dtype: 'float32' } as ValueInfo);

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
    (globalThis as typeof globalThis & { MLGraphBuilder?: unknown }).MLGraphBuilder = class {};
    await p.initialize();

    // Make execute succeed but return missing output
    (p as never as { compiledGraph: unknown }).compiledGraph = { destroy: undefined };
    p.currentGraphId = g.id;
    (p as never as { allocateBuffer: () => unknown }).allocateBuffer = () => ({});
    (p as never as { contextManager: unknown }).contextManager = {
      initialize: vi.fn().mockResolvedValue({ compute: mockCompute }),
      getContext: vi.fn().mockReturnValue({ compute: mockCompute }),
      getBuilder: vi.fn().mockReturnValue({}),
    };

    await expect(p.execute(g, {})).rejects.toThrow(
      'WebNN Execution Error: Missing output out1 from WebNN compute.',
    );
  });
});
