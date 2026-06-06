import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../../src/mmdnn/keras/index';

describe('index.ts', () => {
  it('should load module', () => {
    expect(Module).toBeDefined();
  });
});
