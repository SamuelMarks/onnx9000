import { describe, it, expect } from 'vitest';
import { LegacyQuirkResolver } from '../../../src/mmdnn/legacy/quirks.js';

describe('legacy quirks', () => {
  it('should resolve caffe padding', () => {
    expect(LegacyQuirkResolver.resolveCaffePadding([1])).toEqual([1, 1, 1, 1]);
  });
});
