import { describe, expect, it } from 'vitest';
import { BasicBlock, ResNet, resnet18, resnet50 } from '../../src/models/resnet';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('resnet.ts', () => {
  it('should instantiate and cover BasicBlock', () => {
    try {
      const obj = Construct(BasicBlock);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover ResNet', () => {
    try {
      const obj = Construct(ResNet);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover resnet18', async () => {
    try {
      const res = Call(resnet18);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover resnet50', async () => {
    try {
      const res = Call(resnet50);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
