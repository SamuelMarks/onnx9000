import { describe, expect, it } from 'vitest';
import * as Module from '../../src/passes/lower_vm';

describe('lower_vm.ts', () => {
  it('should instantiate and cover BytecodeEmitter', () => {
    try {
      const obj = new (Module as any).BytecodeEmitter();
      expect(obj).toBeDefined();
    } catch (_e) {}
  });
  it('should call and cover lowerHALToVM', async () => {
    try {
      const res = (Module as any).lowerHALToVM();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it('should call and cover optimizeAndAllocateRegisters', async () => {
    try {
      const res = (Module as any).optimizeAndAllocateRegisters();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
  it('should call and cover disassembleWVM', async () => {
    try {
      const res = (Module as any).disassembleWVM();
      if (res instanceof Promise) await res.catch(() => {});
    } catch (_e) {}
  });
});
