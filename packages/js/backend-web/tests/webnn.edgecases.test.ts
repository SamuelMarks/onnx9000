import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Graph, Node, Tensor, Attribute } from '@onnx9000/core';
import { WebNNCompiler } from '../src/providers/webnn/compiler.js';
import { WebNNContextManager } from '../src/providers/webnn/context.js';
import { WebNNProvider } from '../src/providers/webnn/index.js';

describe('WebNN Context Edge Cases', () => {
  beforeEach(() => {
    WebNNContextManager.getInstance().reset();
  });

  it('should throw if getting context before init', () => {
    expect(() => WebNNContextManager.getInstance().getContext()).toThrow(
      'WebNN Context is not initialized.',
    );
  });

  it('should throw if getting builder before init', () => {
    expect(() => WebNNContextManager.getInstance().getBuilder()).toThrow(
      'WebNN MLGraphBuilder is not initialized.',
    );
  });

  it('should get capabilities if available', async () => {
    const mockLimits = { input: { dataTypes: ['float32'] } };
    Object.defineProperty(global, 'navigator', {
      value: {
        ml: {
          createContext: vi.fn().mockResolvedValue({
            opSupportLimits: () => mockLimits,
          }),
        },
      },
      configurable: true,
    });
    // @ts-ignore
    global.MLGraphBuilder = class {
      constructor() {}
    };

    await WebNNContextManager.getInstance().initialize();
    expect(WebNNContextManager.getInstance().getCapabilities()).toEqual(mockLimits);
  });

  it('should return null capabilities if unavailable', async () => {
    Object.defineProperty(global, 'navigator', {
      value: { ml: { createContext: vi.fn().mockResolvedValue({}) } },
      configurable: true,
    });
    // @ts-ignore
    global.MLGraphBuilder = class {
      constructor() {}
    };
    await WebNNContextManager.getInstance().initialize();
    expect(WebNNContextManager.getInstance().getCapabilities()).toBeNull();
  });

  it('should throw if MLGraphBuilder is totally missing', async () => {
    Object.defineProperty(global, 'navigator', {
      value: { ml: { createContext: vi.fn().mockResolvedValue({}) } },
      configurable: true,
    });
    // @ts-ignore
    global.MLGraphBuilder = undefined;
    (globalThis as any).MLGraphBuilder = undefined;
    await expect(WebNNContextManager.getInstance().initialize()).rejects.toThrow(
      'MLGraphBuilder is not available',
    );
  });
});

describe('WebNNProvider buffer pool and execution edge cases', () => {
  beforeEach(() => {
    WebNNContextManager.getInstance().reset();
  });

  it('should re-use buffers and trigger GC', async () => {
    const mockContext = {
      compute: vi
        .fn()
        .mockImplementation(() =>
          Promise.resolve({ outputs: new Proxy({}, { get: () => new Float32Array([1.0]) }) }),
        ),
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
        return {};
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

    // Create >100 dummy graphs to trigger GC
    for (let i = 0; i < 105; i++) {
      const g = new Graph('g' + i);
      g.inputs.push({ name: 'in', shape: [1], id: 'in', dtype: 'float32' });
      g.nodes.push(new Node('Abs', ['in'], [typeof i !== 'undefined' ? 'out' + i : 'out']));
      g.outputs.push({
        name: typeof i !== 'undefined' ? 'out' + i : 'out',
        shape: [1],
        id: 'out',
        dtype: 'float32',
      } as any);
      await provider.execute(g, {
        in: new Tensor('in', [1], 'float32', false, true, new Float32Array([1])),
      });
    }

    expect(mockDestroy).toHaveBeenCalled(); // graph disposal
    // It should have cleared the pool. We can indirectly see it ran.
  });

  it('should handle different output data types', async () => {
    const mockContext = {
      compute: vi.fn().mockResolvedValue({
        outputs: {
          out16: new Uint16Array([1]),
          out8: new Int8Array([1]),
          out64: new BigInt64Array([1n]),
        },
      }),
    };
    Object.defineProperty(global, 'navigator', {
      value: { ml: { createContext: vi.fn().mockResolvedValue(mockContext) } },
      configurable: true,
    });
    // @ts-ignore
    global.MLGraphBuilder = class {
      input() {
        return {};
      }
      constant() {
        return {};
      }
      build() {
        return Promise.resolve({});
      }
      abs() {
        return {};
      }
      split() {
        return [{}, {}, {}];
      }
    };
    const provider = new WebNNProvider();
    await provider.initialize();

    const g = new Graph('g');
    g.inputs.push({ name: 'in', shape: [1], id: 'in', dtype: 'float32' });
    g.nodes.push(new Node('Split', ['in'], ['out16', 'out8', 'out64']));
    g.outputs.push({ name: 'out16', shape: [1], id: 'o1', dtype: 'float16' } as any);
    g.outputs.push({ name: 'out8', shape: [1], id: 'o2', dtype: 'int8' } as any);
    g.outputs.push({ name: 'out64', shape: [1], id: 'o3', dtype: 'int64' } as any);

    await provider.execute(g, {
      in: new Tensor('in', [1], 'float32', false, true, new Float32Array([1])),
    });
  });

  it('should catch execution errors', async () => {
    const mockContext = { compute: vi.fn().mockRejectedValue(new Error('NPU exploded')) };
    Object.defineProperty(global, 'navigator', {
      value: { ml: { createContext: vi.fn().mockResolvedValue(mockContext) } },
      configurable: true,
    });
    // @ts-ignore
    global.MLGraphBuilder = class {
      input() {
        return {};
      }
      constant() {
        return {};
      }
      build() {
        return Promise.resolve({});
      }
      abs() {
        return {};
      }
      split() {
        return [{}, {}, {}];
      }
    };
    const provider = new WebNNProvider();
    await provider.initialize();
    const g = new Graph('g');
    g.inputs.push({ name: 'in', shape: [1], id: 'i', dtype: 'float32' });
    g.nodes.push(new Node('Abs', ['in'], [typeof i !== 'undefined' ? 'out' + i : 'out']));
    g.outputs.push({
      name: typeof i !== 'undefined' ? 'out' + i : 'out',
      shape: [1],
      id: 'out',
      dtype: 'float32',
    } as any);
    await expect(provider.execute(g, {})).rejects.toThrow('WebNN Execution Error: NPU exploded');
  });

  it('should catch compilation errors', async () => {
    const mockContext = { compute: vi.fn() };
    Object.defineProperty(global, 'navigator', {
      value: { ml: { createContext: vi.fn().mockResolvedValue(mockContext) } },
      configurable: true,
    });
    // @ts-ignore
    global.MLGraphBuilder = class {
      input() {
        return {};
      }
      constant() {
        return {};
      }
      build() {
        return Promise.reject(new Error('Compile exploded'));
      }
      abs() {
        return {};
      }
    };
    const provider = new WebNNProvider();
    await provider.initialize();
    const g = new Graph('g');
    g.inputs.push({ name: 'in', shape: [1], id: 'i', dtype: 'float32' });
    g.nodes.push(new Node('Abs', ['in'], [typeof i !== 'undefined' ? 'out' + i : 'out']));
    g.outputs.push({
      name: typeof i !== 'undefined' ? 'out' + i : 'out',
      shape: [1],
      id: 'out',
      dtype: 'float32',
    } as any);
    await expect(provider.execute(g, {})).rejects.toThrow(
      'WebNN Compilation Error: Compile exploded',
    );
  });
});

describe('WebNNCompiler Edge Cases', () => {
  it('should handle dynamic shapes', async () => {
    // @ts-ignore
    global.MLGraphBuilder = class {
      input = vi.fn().mockReturnValue({});
      build = vi.fn().mockResolvedValue({});
    };
    const builder = new (global as any).MLGraphBuilder();
    const g = new Graph('g');
    g.inputs.push({ name: 'in', shape: ['N'], id: 'in', dtype: 'float32' });
    g.outputs.push({ name: 'in', shape: ['N'], id: 'in', dtype: 'float32' } as any);
    const compiler = new WebNNCompiler(g, builder);
    await compiler.compile();
    expect(builder.input).toHaveBeenCalledWith('in', { dataType: 'float32', dimensions: [1] });
  });

  it('should handle missing inputs gracefully', async () => {
    const builder = new (global as any).MLGraphBuilder();
    const g = new Graph('g');
    g.inputs.push({ name: 'in', shape: [1], id: 'in', dtype: 'float32' });
    // Intentionally omit input mapping for "missing_in"
    g.nodes.push(new Node('Add', ['missing_in'], ['out']));
    const compiler = new WebNNCompiler(g, builder);
    await expect(compiler.compile()).rejects.toThrow(
      'Input operand missing_in not found for node Add',
    );
  });

  it('should throw on unsupported nodes', async () => {
    const builder = new (global as any).MLGraphBuilder();
    const g = new Graph('g');
    g.inputs.push({ name: 'in', shape: [1], id: 'in', dtype: 'float32' });
    g.nodes.push(new Node('NonZero', ['in'], ['out']));
    const compiler = new WebNNCompiler(g, builder);
    builder.input = vi.fn().mockReturnValue({});
    await expect(compiler.compile()).rejects.toThrow('Unsupported WebNN node type: NonZero');
  });

  it('should handle unsupported types', async () => {
    const builder = new (global as any).MLGraphBuilder();
    const g = new Graph('g');
    g.inputs.push({ name: 'in', shape: [1], id: 'in', dtype: 'complex64' as any });
    const compiler = new WebNNCompiler(g, builder);
    await expect(compiler.compile()).rejects.toThrow('Unsupported WebNN data type: complex64');
  });

  it('should map float64 to float32', async () => {
    const builder = new (global as any).MLGraphBuilder();
    const g = new Graph('g');
    g.inputs.push({ name: 'in', shape: [1], id: 'in', dtype: 'float64' });
    g.outputs.push({ name: 'in', shape: [1], id: 'in', dtype: 'float64' } as any);
    const compiler = new WebNNCompiler(g, builder);
    await compiler.compile();
    expect(builder.input).toHaveBeenCalledWith('in', { dataType: 'float32', dimensions: [1] });
  });

  it('should map bool to uint8', async () => {
    const builder = new (global as any).MLGraphBuilder();
    const g = new Graph('g');
    g.inputs.push({ name: 'in', shape: [1], id: 'in', dtype: 'bool' as any });
    g.outputs.push({ name: 'in', shape: [1], id: 'in', dtype: 'bool' } as any);
    const compiler = new WebNNCompiler(g, builder);
    await compiler.compile();
    expect(builder.input).toHaveBeenCalledWith('in', { dataType: 'uint8', dimensions: [1] });
  });

  it('should map int64 to int32', async () => {
    const builder = new (global as any).MLGraphBuilder();
    const g = new Graph('g');
    g.inputs.push({ name: 'in', shape: [1], id: 'in', dtype: 'int64' });
    g.outputs.push({ name: 'in', shape: [1], id: 'in', dtype: 'int64' } as any);
    const compiler = new WebNNCompiler(g, builder);
    await compiler.compile();
    expect(builder.input).toHaveBeenCalledWith('in', { dataType: 'int32', dimensions: [1] });
  });

  it('should map uint64 to uint32', async () => {
    const builder = new (global as any).MLGraphBuilder();
    const g = new Graph('g');
    g.inputs.push({ name: 'in', shape: [1], id: 'in', dtype: 'uint64' });
    g.outputs.push({ name: 'in', shape: [1], id: 'in', dtype: 'uint64' } as any);
    const compiler = new WebNNCompiler(g, builder);
    await compiler.compile();
    expect(builder.input).toHaveBeenCalledWith('in', { dataType: 'uint32', dimensions: [1] });
  });

  it('should trigger zero-dimension exception', async () => {
    const builder = new (global as any).MLGraphBuilder();
    const g = new Graph('g');
    const shapeData = new Int32Array([0]);
    g.tensors['in'] = new Tensor('in', [1], 'int32', false, true, shapeData);
    // Explicitly modify operand mapped to shape [0] to test the internal check
    builder.input = vi.fn().mockReturnValue({ shape: [0, 5] });
    g.inputs.push({ name: 'in', shape: [0, 5], id: 'in', dtype: 'float32' });
    g.nodes.push(new Node('Add', ['in', 'in'], ['out']));
    const compiler = new WebNNCompiler(g, builder);
    await expect(compiler.compile()).rejects.toThrow(
      'Fallback triggered: WebNN native ops currently fail on 0-dimension shapes',
    );
  });

  it('should throw error if output not found', async () => {
    const builder = new (global as any).MLGraphBuilder();
    const g = new Graph('g');
    g.outputs.push({ name: 'missing_out', shape: [1], id: 'o', dtype: 'float32' } as any);
    const compiler = new WebNNCompiler(g, builder);
    await expect(compiler.compile()).rejects.toThrow(
      'Output missing_out not found in WebNN operands.',
    );
  });
});
