// @ts-nocheck
import { describe, it, expect, vi } from 'vitest';
import { BenchmarkPanel } from '../src/components/BenchmarkPanel.js';

describe('BenchmarkPanel', () => {
  it('should render and handle clicks', async () => {
    (global as any).window.pyodide = { runPythonAsync: vi.fn().mockResolvedValue('output') };

    const panel = new BenchmarkPanel();
    document.body.appendChild(panel.element);

    const btns = panel.element.querySelectorAll('button');
    expect(btns.length).toBe(3);

    btns[0]!.click();
    await new Promise((r) => setTimeout(r, 10));
    expect((global as any).window.pyodide.runPythonAsync).toHaveBeenCalled();
  });
});
