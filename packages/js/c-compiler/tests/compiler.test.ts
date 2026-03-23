import { describe, it, expect, vi } from 'vitest';
import { compileOnnxToC, initCompiler } from '../src/index.js';

// Mock Pyodide
vi.mock('pyodide', () => ({
  loadPyodide: async () => ({
    runPython: vi.fn(),
    runPythonAsync: vi.fn(),
  }),
}));

describe('@onnx9000/c-compiler', () => {
  it('should initialize Pyodide instance', async () => {
    const pyodide = await initCompiler();
    expect(pyodide).toBeDefined();
  });

  it('should return compiled mock strings', async () => {
    const result = await compileOnnxToC(new Uint8Array([0, 1, 2]), { prefix: 'test_' });
    expect(result.header).toContain('test_');
    expect(result.source).toContain('test_');
    expect(result.summary).toContain('Memory');
  });
});
