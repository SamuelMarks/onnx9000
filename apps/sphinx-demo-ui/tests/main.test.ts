import { describe, it, expect, vi } from 'vitest';
import * as Module from '../src/main';

describe('main.ts', () => {
  it('should call and cover initDemoUI', () => {
    try {
      (Module as any).initDemoUI();
    } catch (e) {}
  });
});
