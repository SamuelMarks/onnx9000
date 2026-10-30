import { describe, expect, it, vi } from 'vitest';
import { trtFfi } from '../src/ffi.js';
import { Builder, TensorRTProvider } from '../src/index.js';

vi.mock('../src/ffi.js', () => ({
  trtFfi: {
    lib: {
      createInferBuilder_INTERNAL: vi.fn().mockReturnValue({}),
      createNetworkV2: vi.fn().mockReturnValue({}),
      destroyInferBuilder: vi.fn(),
      destroyNetworkDefinition: vi.fn(),
      markOutput: vi.fn(),
    },
    getVersion: vi.fn().mockReturnValue([80, 6, 0]),
  },
}));

describe('TensorRT Builder', () => {
  it('should build and handle errors and repeated destroy', () => {
    const b = new Builder();
    expect(b.ptr).toBeDefined();

    const n = b.createNetwork();
    expect(n.ptr).toBeDefined();

    n.markOutput({ ptr: {} });

    n.destroy();
    n.destroy();
    b.destroy();
    b.destroy();

    const lib = trtFfi.lib;
    if (!lib) throw new Error('Expected trtFfi.lib to be defined');

    vi.mocked(trtFfi.getVersion).mockReturnValueOnce([0, 0, 0]);
    vi.mocked(lib.createInferBuilder_INTERNAL).mockReturnValueOnce(null);
    expect(() => new Builder()).toThrow('Failed to create Builder');

    const b2 = new Builder();
    vi.mocked(lib.createNetworkV2).mockReturnValueOnce(null);
    expect(() => b2.createNetwork()).toThrow('Failed to create NetworkDefinition');

    const origLib = trtFfi.lib;
    trtFfi.lib = null;
    expect(() => new Builder()).toThrow('TensorRT library not loaded');
    trtFfi.lib = origLib;

    const p = new TensorRTProvider({ name: 'test' });
    expect(p.graph).toEqual({ name: 'test' });
  });
});
