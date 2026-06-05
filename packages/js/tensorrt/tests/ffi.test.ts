import { describe, it, expect, vi } from 'vitest';
import { TensorRTFFI } from '../src/ffi.js';

vi.mock('ffi-napi', () => ({
  default: {
    Library: vi.fn().mockReturnValue({ getInferLibVersion: vi.fn().mockReturnValue(80600) }),
  },
}));

describe('TensorRTFFI', () => {
  it('should load', () => {
    const ffi = new TensorRTFFI();
    expect(ffi.getVersion()).toEqual([80, 6, 0]);
  });
});
