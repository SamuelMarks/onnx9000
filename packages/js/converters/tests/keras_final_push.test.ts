import { describe, it, expect, vi } from 'vitest';
import { Keras2OnnxConverter } from '../src/keras/index.js';
import { registerCustomKerasLayer } from '../src/keras/plugin-registry.js';

describe('Keras Final Coverage Push', () => {
  it('should cover masking propagation with Gather embed_masking', async () => {
    // We register a custom layer that emits a Gather node with the magic name
    registerCustomKerasLayer('MaskingLayer', (nodeName, layerName, inputs, outName, config) => {
      return [
        {
          opType: 'Gather',
          inputs: [inputs[0], inputs[0]],
          outputs: [outName],
          name: 'embed_masking_node',
          attributes: [],
        },
      ];
    });

    const modelJson = JSON.stringify({
      format: 'layers-model',
      modelTopology: {
        class_name: 'Sequential',
        config: {
          layers: [
            { class_name: 'InputLayer', config: { batch_input_shape: [null, 10], name: 'in' } },
            { class_name: 'MaskingLayer', config: { name: 'mask' } },
          ],
        },
      },
      weightsManifest: [],
    });

    const converter = new Keras2OnnxConverter(modelJson);
    converter.convert();
    // It should have emitted Constant, Equal, Not nodes for masking
    expect(converter._test_finalNodes.find((n) => n.opType === 'Equal')).toBeDefined();
    expect(converter._test_finalNodes.find((n) => n.opType === 'Not')).toBeDefined();
  });

  it('should cover weight loader gaps and initializers', async () => {
    const modelJson = JSON.stringify({
      format: 'layers-model',
      modelTopology: {
        class_name: 'Sequential',
        config: {
          layers: [
            {
              class_name: 'Dense',
              config: { name: 'd1', units: 10, batch_input_shape: [null, 5] },
            },
          ],
        },
      },
      weightsManifest: [],
    });
    const converter = new Keras2OnnxConverter(modelJson);
    converter.convert();
    // This should hit the initializer logic for _weights, _kernel, _bias
    expect(converter._test_finalNodes.length).toBeGreaterThan(0);
  });
});
