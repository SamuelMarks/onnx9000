import { describe, it, expect } from 'vitest';
import { Keras2OnnxConverter } from '../src/keras/index.js';

describe('Keras2OnnxConverter Extra Gaps V3', () => {
  it('should cover Minimum and Maximum with 3 nodes (variadic)', async () => {
    for (const className of ['Minimum', 'Maximum']) {
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
                class_name: 'InputLayer',
                name: 'in3',
                config: { batch_input_shape: [null, 10], name: 'in3' },
                inbound_nodes: [],
              },
              {
                class_name: className,
                name: 'math',
                config: { name: 'math' },
                inbound_nodes: [
                  [
                    ['in1', 0, 0, {}],
                    ['in2', 0, 0, {}],
                    ['in3', 0, 0, {}],
                  ],
                ],
              },
            ],
            input_layers: [['in1', 0, 0]],
            output_layers: [['math', 0, 0]],
          },
        },
        weightsManifest: [],
      });
      const converter = new Keras2OnnxConverter(modelJson);
      converter.convert();
      const op = className === 'Minimum' ? 'Min' : 'Max';
      const node = converter._test_finalNodes.find((n) => n.opType === op);
      expect(node).toBeDefined();
      expect(node!.inputs.length).toBe(3);
    }
  });
});
