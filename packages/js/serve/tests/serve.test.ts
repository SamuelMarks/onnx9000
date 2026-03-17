import { describe, it, expect } from 'vitest';
import * as serve from '../src/index';

describe('serve', () => {
  it('should export nothing or something', () => {
    expect(serve).toBeDefined();
  });
});
