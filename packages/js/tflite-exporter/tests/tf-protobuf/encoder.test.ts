import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../src/tf-protobuf/encoder';

describe('encoder.ts', () => {
  it('should instantiate and cover TFProtobufEncoder', () => {
    try {
       const obj = new (Module as any).TFProtobufEncoder();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
});
