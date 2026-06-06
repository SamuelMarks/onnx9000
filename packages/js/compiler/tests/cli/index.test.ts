import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../src/cli/index';

describe('index.ts', () => {
  it('should load module', () => {
    expect(Module).toBeDefined();
  });
});
