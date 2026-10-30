import * as os from 'node:os';
import ffi from 'ffi-napi';
import { describe, expect, it, vi } from 'vitest';
import { TensorRTFFI } from '../src/ffi.js';

vi.mock('node:os', async (importOriginal) => {
  const actual = await importOriginal<typeof import('node:os')>();
  return {
    ...actual,
    platform: vi.fn(() => actual.platform()),
  };
});

vi.mock('ffi-napi', () => ({
  default: {
    Library: vi.fn().mockReturnValue({ getInferLibVersion: vi.fn().mockReturnValue(80600) }),
  },
}));

describe('TensorRTFFI', () => {
  it('should load and handle win32, errors, and missing version', () => {
    const f1 = new TensorRTFFI();
    expect(f1.getVersion()).toEqual([80, 6, 0]);

    vi.mocked(os.platform).mockReturnValueOnce('win32');
    vi.mocked(ffi.Library).mockImplementationOnce(() => {
      throw new Error('load failed');
    });
    const f2 = new TensorRTFFI();
    expect(f2.getVersion()).toEqual([0, 0, 0]);
  });
});
