import { describe, expect, it, vi } from 'vitest';
import { validateBlock } from '../src/mil/validator.js';

vi.mock('../src/sort.js', () => ({ topologicalSort: vi.fn() }));

describe('validator', () => {
  it('should validate', () => {
    const block = {
      name: 'test',
      inputs: [{ name: 'in' }],
      outputs: [{ name: 'out' }],
      operations: [
        {
          opType: 'const',
          inputs: { x: { name: 'in' } },
          outputs: [{ name: 'out' }],
        },
      ],
    };
    expect(() =>
      validateBlock(block as never as Parameters<typeof validateBlock>[0]),
    ).not.toThrow();
  });
});
