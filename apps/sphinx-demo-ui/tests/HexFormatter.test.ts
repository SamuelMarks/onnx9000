// @ts-nocheck
import { describe, it, expect } from 'vitest';
import { HexFormatter } from '../src/core/HexFormatter.js';

describe('HexFormatter', () => {
  it('should format hex', () => {
    const arr = new Uint8Array([0x41, 0x42, 0x43]);
    const res = HexFormatter.format(arr);
    expect(res).toContain('41 42 43');
    expect(res).toContain('ABC');
  });
});
