import { describe, it, expect } from 'vitest';
import { extractKerasTopology } from '../src/keras/keras-ast.js';

describe('keras-ast', () => {
  it('should extract topology', () => {
    const top = extractKerasTopology({ class_name: 'Sequential', config: { layers: [] } });
    expect(top.inputs).toBeDefined();
    expect(top.outputs).toBeDefined();
    expect(top.nodes).toBeDefined();
  });
});
