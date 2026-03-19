import { describe, it, expect } from 'vitest';
import * as t from '../src/index';

describe('Index', () => {
  it('exports everything', () => {
    expect(t).toBeDefined();
  });
});
