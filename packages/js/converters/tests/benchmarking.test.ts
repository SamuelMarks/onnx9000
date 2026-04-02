import { describe, it, expect, vi } from 'vitest';
import { ChromeTraceExporter, validateMathematicalTolerance } from '../src/keras/benchmarking.js';
import * as fs from 'fs';

vi.mock('fs');

describe('ChromeTraceExporter', () => {
  it('should record and save events', () => {
    const exporter = new ChromeTraceExporter();
    exporter.startEvent('test_op');
    exporter.recordMemory(1024);
    exporter.endEvent('test_op');

    exporter.save('trace.json');
    expect(fs.writeFileSync).toHaveBeenCalled();
    const callArgs = vi.mocked(fs.writeFileSync).mock.calls[0];
    expect(callArgs[0]).toBe('trace.json');
    const events = JSON.parse(callArgs[1] as string);
    expect(events).toHaveLength(3);
    expect(events[0].name).toBe('test_op');
    expect(events[0].ph).toBe('B');
    expect(events[1].name).toBe('MemoryAlloc');
    expect(events[2].ph).toBe('E');
  });
});

describe('validateMathematicalTolerance', () => {
  it('should return true for values within tolerance', () => {
    const a = new Float32Array([1.0, 2.0]);
    const b = new Float32Array([1.00001, 1.99999]);
    expect(validateMathematicalTolerance(a, b, 1e-4)).toBe(true);
  });

  it('should return false for values outside tolerance', () => {
    const a = new Float32Array([1.0, 2.0]);
    const b = new Float32Array([1.1, 2.0]);
    expect(validateMathematicalTolerance(a, b, 1e-4)).toBe(false);
  });

  it('should return false for mismatched lengths', () => {
    const a = new Float32Array([1.0]);
    const b = new Float32Array([1.0, 2.0]);
    expect(validateMathematicalTolerance(a, b)).toBe(false);
  });
});
