import { describe, it, expect } from 'vitest';
import { TFLiteExporter } from '../src/exporter';

describe('Exporter Extra', () => {
  it('should call builder.clear in destroy if available', () => {
    const exporter = new TFLiteExporter();
    let clearCalled = false;
    // mock builder.clear
    (exporter as Object).builder.clear = () => {
      clearCalled = true;
    };
    exporter.destroy();
    expect(clearCalled).toBe(true);
  });
});
