import { describe, it, expect, vi } from 'vitest';
import {
  registerCustomKerasLayer,
  getCustomKerasLayerEmitter,
} from '../src/keras/plugin-registry.js';
import { Keras2OnnxConverter } from '../src/keras/index.js';

describe('Keras Custom Layer Plugin Registry', () => {
  it('registers and retrieves custom layers', () => {
    const customEmitter = vi.fn().mockReturnValue([]);
    registerCustomKerasLayer('MyCustomLayer', customEmitter);

    const retrieved = getCustomKerasLayerEmitter('MyCustomLayer');
    expect(retrieved).toBe(customEmitter);
  });

  it('triggers custom emitter during conversion', () => {
    const customEmitter = vi.fn().mockReturnValue([
      {
        opType: 'CustomOnnxOp',
        name: 'custom_node',
        inputs: ['in_custom'],
        outputs: ['out_custom'],
        attributes: [],
      },
    ]);
    registerCustomKerasLayer('MagicalAttention', customEmitter);

    const modelJson = JSON.stringify({
      format: 'layers-model',
      weightsManifest: [],
      modelTopology: {
        class_name: 'Sequential',
        config: {
          layers: [
            {
              class_name: 'InputLayer',
              config: { name: 'in1', batch_input_shape: [null, 10], dtype: 'float32' },
            },
            { class_name: 'MagicalAttention', config: { name: 'MagicalAttention', num_heads: 8 } },
          ],
        },
      },
    });

    const converter = new Keras2OnnxConverter(modelJson);
    converter.convert();

    expect(customEmitter).toHaveBeenCalled();
    const callArgs = customEmitter.mock.calls[0] as Object[];
    expect(callArgs[0]).toBe('MagicalAttention:0'); // nodeName
    expect(callArgs[4]).toEqual({ name: 'MagicalAttention', num_heads: 8 }); // config
  });
});
