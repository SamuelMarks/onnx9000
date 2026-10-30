import { describe, expect, it, vi } from 'vitest';
import { KerasPythonParser } from '../src/core/KerasPythonParser.js';

describe('KerasPythonParser', () => {
  it('should init pyodide and parse', async () => {
    const win = window as Window & { loadPyodide?: ReturnType<typeof vi.fn> };
    win.loadPyodide = vi.fn().mockResolvedValue({
      runPythonAsync: vi.fn().mockResolvedValue('{"modelTopology": {}}'),
    });

    const res = (await KerasPythonParser.parse('import keras')) as { modelTopology?: unknown };
    expect(res).toBeDefined();
    expect(res.modelTopology).toBeDefined();
  });
});
