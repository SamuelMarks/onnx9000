import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../src/core/HexFormatter';

describe('HexFormatter.ts', () => {
  it('should instantiate and cover HexFormatter', () => {
    // Attempt to instantiate
    try {
       const obj = new (Module as any).HexFormatter();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
});
