import { describe, it, expect } from 'vitest';
import { ONNXToMILConverter } from '../src/converter.js';
import { Graph } from '@onnx9000/core';

describe('ONNXToMILConverter', () => {
  it('handles various ONNX nodes', () => {
    const graph: Graph = {
      name: 'test',
      inputs: [
        { name: 'in1', dtype: 'float32', shape: [1] },
        { name: 'in2', dtype: 'int32', shape: [1] },
        { name: 'in3', dtype: 'bool', shape: [1] },
        { name: 'in4', dtype: 'float64', shape: [1] }, // Should warn downcast
        { name: 'in5', dtype: 'float16', shape: [1] },
        { name: 'in6', dtype: 'int64', shape: [1] },
      ],
      outputs: [{ name: 'out1', dtype: 'float32', shape: [1] }],
      initializers: ['init1'],
      tensors: {
        init1: { dtype: 'float32', shape: [1], data: new Float32Array([1]) },
      },
      nodes: [
        { name: 'n', opType: 'Add', inputs: ['in1', 'init1'], outputs: ['t1'], attributes: {} },
        {
          name: 'n',
          opType: 'Pad',
          inputs: ['t1'],
          outputs: ['t2'],
          attributes: { pads: { type: 'INTS', value: [1, 1] } },
        },
        {
          name: 'n',
          opType: 'Conv',
          inputs: ['t2'],
          outputs: ['t3'],
          attributes: {
            group: { type: 'INT', value: 2 },
            auto_pad: { type: 'STRING', value: 'SAME_UPPER' },
          },
        },
        {
          name: 'n',
          opType: 'Conv',
          inputs: ['t3'],
          outputs: ['t4'],
          attributes: { auto_pad: { type: 'STRING', value: 'VALID' } },
        },
        {
          name: 'n',
          opType: 'Resize',
          inputs: ['t4'],
          outputs: ['t5'],
          attributes: {
            coordinate_transformation_mode: { type: 'STRING', value: 'align_corners' },
            mode: { type: 'STRING', value: 'linear' },
          },
        },
        {
          name: 'n',
          opType: 'Resize',
          inputs: ['t5'],
          outputs: ['t6'],
          attributes: {
            coordinate_transformation_mode: { type: 'STRING', value: 'half_pixel' },
            mode: { type: 'STRING', value: 'nearest' },
          },
        },
        {
          name: 'n',
          opType: 'LSTM',
          inputs: ['t6'],
          outputs: ['t7'],
          attributes: {
            direction: { type: 'STRING', value: 'BIDIRECTIONAL' },
            layout: { type: 'INT', value: 0 },
          },
        },
        { name: 'n', opType: 'LSTM', inputs: ['t7'], outputs: ['t8'], attributes: {} },
        {
          name: 'n',
          opType: 'Identity',
          inputs: ['t8'],
          outputs: ['t9'],
          attributes: { keepdims: { type: 'INT', value: 1 } },
        },
        { name: 'n', opType: 'If', inputs: ['in3'], outputs: ['out1'], attributes: {} },
        { name: 'n', opType: 'Loop', inputs: ['in2'], outputs: ['out2'], attributes: {} },
        {
          name: 'n',
          opType: 'LayerNormalization',
          inputs: ['in2'],
          outputs: ['out3'],
          attributes: {},
        },
        {
          name: 'n',
          opType: 'InstanceNormalization',
          inputs: ['in2'],
          outputs: ['out4'],
          attributes: {},
        },
        {
          name: 'n',
          opType: 'BatchNormalization',
          inputs: ['in2'],
          outputs: ['out5'],
          attributes: {},
        },
      ],
    };

    // Need out2 mapping so it doesnt fail on output mapping?
    // Let's change outputs to match.
    graph.outputs = [
      { name: 'out1', dtype: 'float32', shape: [1] },
      { name: 'out2', dtype: 'float32', shape: [1] },
      { name: 'out3', dtype: 'float32', shape: [1] },
      { name: 'out4', dtype: 'float32', shape: [1] },
      { name: 'out5', dtype: 'float32', shape: [1] },
    ];

    const converter = new ONNXToMILConverter(graph, { dynamicBatching: true });

    // Stub process.env to trigger telemetry warning
    const oldEnv = process.env['ONNX9000_TELEMETRY'];
    process.env['ONNX9000_TELEMETRY'] = '1';

    const program = converter.convert();
    expect(program).toBeDefined();

    process.env['ONNX9000_TELEMETRY'] = oldEnv;
  });

  it('throws on unsupported op', () => {
    const graph: Graph = {
      name: 'fail',
      inputs: [{ name: 'in1', dtype: 'float32', shape: [1] }],
      outputs: [{ name: 'out1', dtype: 'float32', shape: [1] }],
      initializers: [],
      tensors: {},
      nodes: [
        {
          name: 'n',
          opType: 'SuperUnsupportedFakeOp9000',
          inputs: ['in1'],
          outputs: ['out1'],
          attributes: {},
        },
      ],
    };
    const converter = new ONNXToMILConverter(graph);
    expect(() => converter.convert()).toThrowError('Operation is completely unsupported');
  });

  it('throws on missing output', () => {
    const graph: Graph = {
      name: 'fail',
      inputs: [{ name: 'in1', dtype: 'float32', shape: [1] }],
      outputs: [{ name: 'missing_out', dtype: 'float32', shape: [1] }],
      initializers: [],
      tensors: {},
      nodes: [
        { name: 'n', opType: 'Identity', inputs: ['in1'], outputs: ['out1'], attributes: {} },
      ],
    };
    const converter = new ONNXToMILConverter(graph);
    expect(() => converter.convert()).toThrowError('not found in MIL graph');
  });
});
