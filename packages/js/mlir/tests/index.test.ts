import { describe, expect, it } from 'vitest';
import { Mlir } from '../src/index.js';

describe('Mlir', () => {
  it('should build and emit textual MLIR dialect operations', () => {
    const mlir = new Mlir();
    expect(mlir.run()).toBe('[mlir] processed');
    const text = mlir.emitModule('forward');
    expect(text).toContain('func.func @forward()');
    expect(text).toContain('"tosa.add"(%arg0, %arg1) : () -> tensor<4xf32>');
  });
});
