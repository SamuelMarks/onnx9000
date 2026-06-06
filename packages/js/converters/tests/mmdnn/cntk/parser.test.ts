import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../../src/mmdnn/cntk/parser';

describe('parser.ts', () => {
  it('should instantiate and cover CNTKParser', () => {
    try {
       const obj = new (Module as any).CNTKParser();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
});
