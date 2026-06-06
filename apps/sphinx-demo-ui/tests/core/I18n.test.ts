import { describe, expect, it } from 'vitest';
import * as Module from '../../src/core/I18n';

describe('I18n.ts', () => {
  it('should load module', () => {
    expect(Module).toBeDefined();
  });
});
