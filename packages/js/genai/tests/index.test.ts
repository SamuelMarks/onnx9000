import { describe, it, expect } from 'vitest';
import { Genai } from '../src/index.js';

describe('Genai', () => {
  it('should run', () => {
    expect(new Genai().run()).toBeDefined();
  });
});
