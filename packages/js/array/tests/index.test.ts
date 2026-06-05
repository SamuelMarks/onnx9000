import { describe, it, expect } from 'vitest';
import { array, add, nn } from '../src/index.js';

describe('array', () => {
  it('should create eager tensor', () => {
    const a = array([1, 2, 3]);
    expect(a).toBeDefined();

    const b = add(a, 1);
    expect(b).toBeDefined();

    const c = nn.relu(a);
    expect(c).toBeDefined();
  });
});
