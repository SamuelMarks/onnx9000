import { describe, it, expect } from 'vitest';
import { emitModel } from '../src/emitter.js';

describe('emitter', () => {
  it('should emit model', () => {
    const bytes = emitModel({ specificationVersion: 1 } as any);
    expect(bytes).toBeDefined();
    expect(bytes.length).toBeGreaterThan(0);
  });
});
