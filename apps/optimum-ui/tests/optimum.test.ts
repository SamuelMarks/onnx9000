import { describe, it, expect } from 'vitest';
import * as opt from '../src/index';

describe('optimum-ui', () => {
  it('should export nothing or something', () => {
    expect(opt).toBeDefined();
  });
});
