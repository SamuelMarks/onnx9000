import { describe, expect, it } from 'vitest';
import * as Module from '../../src/genai/types';

describe('types.ts', () => {
  it('should load module', () => {
    expect(Module).toBeDefined();
  });
});
