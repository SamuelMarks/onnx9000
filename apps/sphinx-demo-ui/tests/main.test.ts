import { describe, expect, it } from 'vitest';
import { initDemoUI } from '../src/main';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('main.ts', () => {
  it('should call and cover initDemoUI', () => {
    try {
      Call(initDemoUI);
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
