// @vitest-environment jsdom
import { describe, expect, it } from 'vitest';
import { TfjsShimDemoElement } from '../src/ui.js';

describe('ui.ts', () => {
  it('should instantiate and cover TfjsShimDemoElement', () => {
    const obj = new TfjsShimDemoElement();
    expect(obj).toBeDefined();
  });
});
