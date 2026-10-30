import { describe, expect, it } from 'vitest';
import { Onnxscript } from '../src/index.js';

describe('Onnxscript', () => {
  it('should record ONNX script operations with SSA outputs', () => {
    const script = new Onnxscript();
    expect(script.run()).toBe('[onnx-script] processed');
    expect(script.nodes).toEqual([
      { opType: 'Add', inputs: ['x', 'b'], output: '%v0' },
      { opType: 'Relu', inputs: ['%v0'], output: '%v1' },
    ]);
  });
});
