import { describe, it, expect } from 'vitest';
import { convert } from '../src/index';

describe('converters', () => {
  it('should convert', () => {
    expect(convert('model')).toBe('converted_model');
  });
});
