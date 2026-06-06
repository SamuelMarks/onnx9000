import { describe, expect, it } from 'vitest';
import { NewModelArch } from '../src/index.js';

describe('NewModelArch', () => {
  it('should run', () => {
    expect(new NewModelArch().process('test')).toContain('test');
  });
});
