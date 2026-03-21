import { describe, it, expect } from 'vitest';
import { sanitizeMetadataString, sanitizeFilename } from '../src/utils/sanitize.js';

describe('Sanitization utilities', () => {
  it('Removes null terminators from metadata strings', () => {
    expect(sanitizeMetadataString('valid\0string')).toBe('validstring');
    expect(sanitizeMetadataString('multiple\0null\0\0terminators')).toBe('multiplenullterminators');
  });

  it('Removes invalid unicode replacement characters', () => {
    expect(sanitizeMetadataString('bad\uFFFDchar')).toBe('badchar');
    expect(sanitizeMetadataString('\uFFFE\uFFFF')).toBe('');
  });

  it('Handles undefined inputs gracefully', () => {
    expect(sanitizeMetadataString(undefined)).toBeUndefined();
    // @ts-expect-error Testing invalid runtime input
    expect(sanitizeMetadataString(123)).toBe(123);
  });

  it('Sanitizes filenames by replacing illegal characters with underscores', () => {
    expect(sanitizeFilename('weight.bin')).toBe('weight.bin');
    expect(sanitizeFilename('my/weird\\file:name.bin')).toBe('my_weird_file_name.bin');
    expect(sanitizeFilename('!@#$%^&*()')).toBe('__________');
    expect(sanitizeFilename('valid_name-1.0.bin')).toBe('valid_name-1.0.bin');
  });
});
