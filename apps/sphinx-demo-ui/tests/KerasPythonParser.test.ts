// @ts-nocheck
import { describe, expect, it, vi } from 'vitest';
import { KerasPythonParser } from '../src/core/KerasPythonParser.js';

describe('KerasPythonParser', () => {
  it('should init pyodide and parse', async () => {
    (global as any).window.loadPyodide = vi.fn().mockResolvedValue({
      runPythonAsync: vi.fn().mockResolvedValue('{"modelTopology": {}}'),
    });

    const res = await KerasPythonParser.parse('import keras');
    expect(res).toBeDefined();
    expect((res as any).modelTopology).toBeDefined();
  });
});
