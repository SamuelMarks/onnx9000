import { describe, expect, it } from 'vitest';
import { Tensor } from '../src/ir/tensor.js';
import { whisperTiny } from '../src/models/whisper.js';

describe('Whisper', () => {
  it('should create and call', () => {
    const model = whisperTiny();
    expect(model).toBeDefined();
    const out = model.call(new Tensor('x', [1], 'float32'), new Tensor('tokens', [1], 'int64'));
    expect(out).toBeDefined();
  });
});
