import { describe, it, expect, vi } from 'vitest';
import { Keras2OnnxConverter } from '../src/keras/index.js';
import { registerCustomKerasLayer } from '../src/keras/plugin-registry.js';

describe('Keras2OnnxConverter Extra Gaps V2', () => {
  it('should cover signature mapping, mixed precision, and complex shape mapping', async () => {
    const modelJson = JSON.stringify({
      format: 'layers-model',
      modelTopology: {
        class_name: 'Functional',
        config: {
          name: 'model',
          layers: [
            {
              class_name: 'InputLayer',
              name: 'in',
              config: { batch_input_shape: [null, 10, null, 5], name: 'in' },
              inbound_nodes: [],
            },
            {
              class_name: 'Add',
              name: 'add',
              config: { name: 'add', dtype: 'float16' },
              inbound_nodes: [[['in', 0, 0, {}]]],
            },
          ],
          input_layers: [['in', 0, 0]],
          output_layers: [['add', 0, 0]],
        },
      },
      signature: {
        serving_default: {
          inputs: { input_sig: { name: 'in:0:0' } },
          outputs: { output_sig: { name: 'add:0:0' } },
        },
      },
      weightsManifest: [],
    });

    const converter = new Keras2OnnxConverter(modelJson);
    converter.convert();
    expect(converter._test_finalNodes.find((n) => n.opType === 'Cast')).toBeDefined();
    expect(converter._test_finalNodes.find((n) => n.outputs.includes('output_sig'))).toBeDefined();
  });

  it('should cover Multiply/Average multi-node and masking', async () => {
    // Register custom layer to emit Gather with magic name
    registerCustomKerasLayer(
      'MaskingLayerFinal',
      (nodeName, layerName, inputs, outName, config) => {
        return [
          {
            opType: 'Gather',
            inputs: [inputs[0], inputs[0]],
            outputs: [outName],
            name: 'embed_masking_node',
            attributes: [],
          },
        ];
      },
    );

    const modelJson = JSON.stringify({
      format: 'layers-model',
      modelTopology: {
        class_name: 'Functional',
        config: {
          name: 'model',
          layers: [
            {
              class_name: 'InputLayer',
              name: 'in1',
              config: { batch_input_shape: [null, 10], name: 'in1' },
              inbound_nodes: [],
            },
            {
              class_name: 'InputLayer',
              name: 'in2',
              config: { batch_input_shape: [null, 10], name: 'in2' },
              inbound_nodes: [],
            },
            {
              class_name: 'Multiply',
              name: 'mul',
              config: { name: 'mul' },
              inbound_nodes: [
                [
                  ['in1', 0, 0, {}],
                  ['in2', 0, 0, {}],
                ],
              ],
            },
            {
              class_name: 'Average',
              name: 'avg',
              config: { name: 'avg' },
              inbound_nodes: [
                [
                  ['in1', 0, 0, {}],
                  ['in2', 0, 0, {}],
                ],
              ],
            },
            {
              class_name: 'MaskingLayerFinal',
              name: 'mask',
              config: { name: 'mask' },
              inbound_nodes: [[['avg', 0, 0, {}]]],
            },
          ],
          input_layers: [['in1', 0, 0]],
          output_layers: [['mask', 0, 0]],
        },
      },
      weightsManifest: [],
    });
    const converter = new Keras2OnnxConverter(modelJson);
    converter.convert();
    expect(converter._test_finalNodes.find((n) => n.opType === 'Mean')).toBeDefined();
    expect(converter._test_finalNodes.find((n) => n.opType === 'Equal')).toBeDefined();
  });
});
