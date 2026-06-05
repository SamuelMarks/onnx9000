import { describe, it, expect } from 'vitest';
import { emitActivation, emitDense, emitIdentity } from '../src/keras/emitters.js';

describe('emitters', () => {
  it('should emit activation', () => {
    let nodes = emitActivation('relu', 'in', 'out', 'n');
    expect(nodes[0].opType).toBe('Relu');

    nodes = emitActivation('swish', 'in', 'out', 'n');
    expect(nodes.length).toBe(2);
    expect(nodes[0].opType).toBe('Sigmoid');
  });

  it('should emit dense', () => {
    const nodes = emitDense('in', 'out', 'w', 'b', 'relu', 'n');
    expect(nodes.length).toBe(3); // MatMul, Add, Relu
    expect(nodes[0].opType).toBe('MatMul');
  });

  it('should emit identity', () => {
    const nodes = emitIdentity('in', 'out', 'n');
    expect(nodes[0].opType).toBe('Identity');
  });
});
