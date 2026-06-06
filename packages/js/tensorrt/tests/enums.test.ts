import { describe, it, expect, vi } from 'vitest';
import * as Module from '../src/enums';

describe('enums.ts', () => {
  it('should load module', () => {
    expect(Module).toBeDefined();
  });
});
