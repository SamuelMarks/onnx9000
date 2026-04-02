import { describe, it, expect } from 'vitest';
import { extractKerasTopology } from '../src/keras/keras-ast.js';

describe('keras-ast nested models', () => {
  it('should flatten nested Functional models into a single topology', () => {
    const config = {
      class_name: 'Functional',
      config: {
        input_layers: [['in1', 0, 0]],
        output_layers: [['nested_model', 0, 0]],
        layers: [
          {
            class_name: 'InputLayer',
            name: 'in1',
            config: { batch_input_shape: [null, 10], dtype: 'float32' },
            inbound_nodes: [],
          },
          {
            class_name: 'Functional', // A nested model
            name: 'nested_model',
            config: {
              input_layers: [['nested_in', 0, 0]],
              output_layers: [['dense_nested', 0, 0]],
              layers: [
                {
                  class_name: 'InputLayer',
                  name: 'nested_in',
                  config: { batch_input_shape: [null, 10], dtype: 'float32' },
                  inbound_nodes: [],
                },
                {
                  class_name: 'Dense',
                  name: 'dense_nested',
                  config: {},
                  inbound_nodes: [[['nested_in', 0, 0, {}]]],
                },
              ],
            },
            inbound_nodes: [[['in1', 0, 0, {}]]],
          },
        ],
      },
    };

    const topology = extractKerasTopology(config as any);

    expect(topology.inputs).toHaveLength(1);
    expect(topology.inputs[0].name).toBe('in1:0:0');
    expect(topology.outputs).toHaveLength(1);
    // The output of the parent should correctly route to the last output of the nested model
    expect(topology.outputs[0].name).toBe('nested_model/dense_nested:0:0');

    expect(topology.nodes.has('in1:0')).toBe(true);
    // The nested InputLayer should be skipped or wired correctly, and internal layers prefixed
    expect(topology.nodes.has('nested_model/dense_nested:0')).toBe(true);

    // The dense layer inside the nested model should now receive input directly from 'in1:0:0'
    // instead of 'nested_in:0:0' (which was the internal input layer of the nested model)
    const nestedDenseNode = topology.nodes.get('nested_model/dense_nested:0')!;
    expect(nestedDenseNode.inboundNodes).toEqual(['in1:0:0']);
  });

  it('should handle nested Sequential models', () => {
    const config = {
      class_name: 'Sequential',
      config: {
        name: 'main_seq',
        layers: [
          {
            class_name: 'InputLayer',
            config: { batch_input_shape: [null, 5], dtype: 'float32', name: 'input1' },
          },
          {
            class_name: 'Sequential',
            config: {
              name: 'inner_seq',
              layers: [
                {
                  class_name: 'Dense',
                  config: { name: 'inner_dense', units: 10 },
                },
              ],
            },
          },
        ],
      },
    };

    const topology = extractKerasTopology(config as any);
    expect(topology.nodes.has('inner_seq/inner_dense:0')).toBe(true);
    expect(topology.inputs[0].name).toBe('input1_input:0:0');
  });
});
