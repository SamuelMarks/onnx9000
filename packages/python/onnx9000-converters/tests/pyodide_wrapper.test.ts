import { describe, expect, it } from 'vitest';
import { pyodideWrapper } from '../src/onnx9000/converters/frontend/pyodide_wrapper.js';

describe('pyodide_wrapper', () => {
  it('should run', () => {
    expect(pyodideWrapper.run()).toBe(true);
  });
});
