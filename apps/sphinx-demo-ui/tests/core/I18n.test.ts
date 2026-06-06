import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../src/core/I18n';

describe('I18n.ts', () => {
  it('should load module', () => {
    expect(Module).toBeDefined();
  });
});
