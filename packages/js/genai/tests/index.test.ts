import { describe, it, expect } from 'vitest';
import { Genai } from '../src/index.js';
describe('genai', () => {
  it('runs', () => {
    expect(new Genai().run()).toBe('[genai] processed');
  });
});
