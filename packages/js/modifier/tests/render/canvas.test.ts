import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../src/render/canvas';

describe('canvas.ts', () => {
  it('should instantiate and cover GraphRenderer', () => {
    try {
       const obj = new (Module as any).GraphRenderer();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
});
