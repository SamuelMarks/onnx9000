import { describe, it, expect } from 'vitest';
import { KerasToMLIRCompiler } from '../src/keras_mlir.js';

describe('KerasToMLIRCompiler', () => {
  it('covers all emits', () => {
    const comp = new KerasToMLIRCompiler();
    expect(
      comp.emitTosaConv2D('in', 'w', 'b', { padding: [1, 1], strides: [1, 1], dilations: [1, 1] }),
    ).toContain('tosa.conv2d');
    expect(comp.emitLinalgDense('in', 'w', 'b')).toContain('linalg.matmul');
    expect(comp.emitScfIf('c', 'true', 'false')).toContain('scf.if');
    expect(comp.emitScfWhile('c', 'b')).toContain('scf.while');
  });
});
