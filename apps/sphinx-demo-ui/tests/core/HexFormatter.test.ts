import { describe, it, expect } from 'vitest';
import { HexFormatter } from '../../src/core/HexFormatter';

describe('HexFormatter', () => {
  it('should format an empty array', () => {
    const result = HexFormatter.format(new Uint8Array([]));
    expect(result).toBe('');
  });

  it('should format a small Uint8Array correctly', () => {
    const data = new Uint8Array([0x48, 0x65, 0x6c, 0x6c, 0x6f]); // 'Hello'
    const result = HexFormatter.format(data);
    // Should contain the hex codes and the ASCII 'Hello'
    expect(result).toContain('48 65 6c 6c 6f');
    expect(result).toContain('|Hello');
  });

  it('should pad short rows properly', () => {
    const data = new Uint8Array([0x01, 0x02]);
    const result = HexFormatter.format(data);
    expect(result).toContain('01 02 ');
    expect(result).toContain('|..');
    // Ensure total string is properly spaced for a full 16-byte row
    expect(result.length).toBeGreaterThan(60);
  });

  it('should format exactly 16 bytes correctly', () => {
    const data = new Uint8Array(16);
    data.fill(0x41); // 'A'
    const result = HexFormatter.format(data);
    expect(result).toContain('41 41 41 41 41 41 41 41  41 41 41 41 41 41 41 41');
    expect(result).toContain('|AAAAAAAAAAAAAAAA|');
  });

  it('should handle unprintable ascii as dots', () => {
    const data = new Uint8Array([0x00, 0x1f, 0x7f, 0xff]); // Non-printables
    const result = HexFormatter.format(data);
    expect(result).toContain('|....');
  });

  it('should truncate outputs exceeding maxLength', () => {
    const data = new Uint8Array(100);
    data.fill(0xff); // Dummy data

    // Request only 32 bytes max
    const result = HexFormatter.format(data, 32);

    // There should be exactly 2 rows of hex (16 bytes each)
    const lines = result.split('\n');

    expect(lines.length).toBeGreaterThan(3); // 2 rows + empty + truncation notice
    expect(result).toContain('truncated');
    expect(result).toContain('first 32 bytes of 100 total');
  });
});
