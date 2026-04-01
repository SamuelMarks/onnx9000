import { describe, it, expect } from 'vitest';
import { extractKerasTopology } from '../src/keras/keras-ast.js';

describe('keras-ast', () => {
  it('should extract Sequential topology', () => {
    const config = {
      class_name: 'Sequential',
      config: {
        layers: [
          {
            class_name: 'InputLayer',
            config: { name: 'in1', batch_input_shape: [null, 10], dtype: 'float32' },
          },
          {
            class_name: 'Dense',
            config: { name: 'dense1' },
          },
        ],
      },
    };

    const topology = extractKerasTopology(config as any);
    expect(topology.inputs).toHaveLength(1);
    expect(topology.inputs[0].name).toBe('in1_input:0:0');
    expect(topology.inputs[0].shape).toEqual([null, 10]);
    expect(topology.outputs).toHaveLength(1);
    expect(topology.outputs[0].name).toBe('dense1:0:0');
    expect(topology.nodes.has('in1:0')).toBe(true);
    expect(topology.nodes.has('dense1:0')).toBe(true);
    expect(topology.nodes.get('dense1:0')?.inboundNodes).toEqual(['in1:0:0']);
  });

  it('should extract Functional topology', () => {
    const config = {
      class_name: 'Functional',
      config: {
        input_layers: [['in1', 0, 0]],
        output_layers: [['dense1', 0, 0]],
        layers: [
          {
            class_name: 'InputLayer',
            name: 'in1',
            config: { batch_input_shape: [null, 5], dtype: 'int32' },
            inbound_nodes: [],
          },
          {
            class_name: 'Dense',
            name: 'dense1',
            config: {},
            inbound_nodes: [[['in1', 0, 0, {}]]],
          },
        ],
      },
    };

    const topology = extractKerasTopology(config as any);
    expect(topology.inputs).toHaveLength(1);
    expect(topology.inputs[0].name).toBe('in1:0:0');
    expect(topology.inputs[0].shape).toEqual([null, 5]);
    expect(topology.inputs[0].dtype).toBe('int32');
    expect(topology.outputs).toHaveLength(1);
    expect(topology.outputs[0].name).toBe('dense1:0:0');
    expect(topology.nodes.has('dense1:0')).toBe(true);
    expect(topology.nodes.get('dense1:0')?.inboundNodes).toEqual(['in1:0:0']);
  });

  it('should handle shared layers in Functional topology', () => {
    const config = {
      class_name: 'Functional',
      config: {
        input_layers: [['in1', 0, 0]],
        output_layers: [['dense1', 0, 0], ['dense1', 1, 0]],
        layers: [
          {
            class_name: 'InputLayer',
            name: 'in1',
            config: { batch_input_shape: [null, 5], dtype: 'float32' },
            inbound_nodes: [],
          },
          {
            class_name: 'Dense',
            name: 'dense1',
            config: {},
            inbound_nodes: [
              [['in1', 0, 0, {}]],
              [['in1', 0, 0, {}]] // Re-used!
            ],
          },
        ],
      },
    };

    const topology = extractKerasTopology(config as any);
    expect(topology.nodes.has('dense1:0')).toBe(true);
    expect(topology.nodes.has('dense1:1')).toBe(true);
    expect(topology.nodes.get('dense1:0')?.inboundNodes).toEqual(['in1:0:0']);
    expect(topology.nodes.get('dense1:1')?.inboundNodes).toEqual(['in1:0:0']);
  });

  it('should throw on unknown topology', () => {
    expect(() => extractKerasTopology({ class_name: 'Unknown', config: {} } as any)).toThrow(
      'Unsupported root model class: Unknown',
    );
  });
});
