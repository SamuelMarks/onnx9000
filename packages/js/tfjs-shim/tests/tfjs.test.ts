import { describe, it, expect } from 'vitest';
import * as tfjs from '../src/index';

describe('tfjs-shim', () => {
  it('should export nothing or something', () => {
    expect(tfjs).toBeDefined();
  });
});
