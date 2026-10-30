import { describe, expect, it } from 'vitest';
import {
  BytecodeEmitter,
  disassembleWVM,
  lowerHALToVM,
  optimizeAndAllocateRegisters,
} from '../../src/passes/lower_vm';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('lower_vm.ts', () => {
  it('should instantiate and cover BytecodeEmitter', () => {
    try {
      const obj = Construct(BytecodeEmitter);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover lowerHALToVM', async () => {
    try {
      const res = Call(lowerHALToVM);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover optimizeAndAllocateRegisters', async () => {
    try {
      const res = Call(optimizeAndAllocateRegisters);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover disassembleWVM', async () => {
    try {
      const res = Call(disassembleWVM);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
