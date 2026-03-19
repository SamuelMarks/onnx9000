import { describe, it, expect } from 'vitest';
import { generateTraceVisualizer, generateHTMLReport, compileInBrowserWorker } from '../src/cli.js';

describe('Full Parity & Future Hardening', () => {
  it('should validate advanced visualizer reports', () => {
    expect(generateTraceVisualizer({})).toContain('HAL Command Buffer Trace');
    expect(generateHTMLReport(['shader'], [])).toContain('WGSL to ONNX Mapping');
  });

  it('should run compiler in worker', async () => {
    const buffer = new ArrayBuffer(10);
    const result = await compileInBrowserWorker(buffer, {
      targetBackend: 'webnn',
      dumpMlir: false,
      optimizeLevel: 'O2',
    });
    expect(result.byteLength).toBe(0);
  });
});
