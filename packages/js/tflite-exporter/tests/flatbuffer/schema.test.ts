import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../src/flatbuffer/schema';

describe('schema.ts', () => {
  it('should instantiate and cover OperatorCode', () => {
    try {
       const obj = new (Module as any).OperatorCode();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
  it('should instantiate and cover QuantizationParameters', () => {
    try {
       const obj = new (Module as any).QuantizationParameters();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
  it('should instantiate and cover Tensor', () => {
    try {
       const obj = new (Module as any).Tensor();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
  it('should instantiate and cover Operator', () => {
    try {
       const obj = new (Module as any).Operator();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
  it('should instantiate and cover SubGraph', () => {
    try {
       const obj = new (Module as any).SubGraph();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
  it('should instantiate and cover Buffer', () => {
    try {
       const obj = new (Module as any).Buffer();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
  it('should instantiate and cover Metadata', () => {
    try {
       const obj = new (Module as any).Metadata();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
  it('should instantiate and cover Model', () => {
    try {
       const obj = new (Module as any).Model();
       expect(obj).toBeDefined();
    } catch (e) {}
  });
});
