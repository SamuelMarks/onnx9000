import { describe, it, expect, vi } from 'vitest';
import { Keras2OnnxConverter } from '../../src/keras/index.js';

vi.mock('../../src/keras/tfjs-parser.js', () => ({
  parseTFJSModel: vi.fn().mockReturnValue({
    format: 'layers-model',
    modelTopology: { class_name: 'Sequential', config: { layers: [] } },
  }),
}));

describe('Keras2OnnxConverter', () => {
  it('should init and convert', () => {
    const conv = new Keras2OnnxConverter('{}');
    const res = conv.convert();
    expect(res.byteLength).toBeGreaterThan(0);
    expect(conv._test_finalNodes).toBeDefined();
  });
});
