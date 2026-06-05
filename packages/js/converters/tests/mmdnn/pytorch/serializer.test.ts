import { describe, it, expect, vi } from 'vitest';
import { PyTorchSerializer } from '../src/mmdnn/pytorch/serializer.js';

vi.mock('fflate', () => ({ zipSync: vi.fn().mockReturnValue(new Uint8Array()) }));

describe('PyTorchSerializer', () => {
  it('should serialize', () => {
    const res = PyTorchSerializer.serialize([
      { name: 'test', shape: [1], dtype: 'float32', size: 1 } as any,
    ]);
    expect(res).toBeDefined();
  });
});
