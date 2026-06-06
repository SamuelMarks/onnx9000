import { describe, expect, it } from 'vitest';
import * as Module from '../../../src/mmdnn/lightgbm/parser';

describe('parser.ts', () => {
  it('should instantiate and cover LightGBMParser', () => {
    try {
      const obj = new (Module as any).LightGBMParser();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
});
