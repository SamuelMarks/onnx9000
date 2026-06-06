import { describe, expect, it } from 'vitest';
import { Graphsurgeon } from '../src/index.js';

describe('Graphsurgeon', () => {
  it('should run', () => {
    expect(new Graphsurgeon().run()).toBeDefined();
  });
});
