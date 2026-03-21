import { describe, it, expect, vi } from 'vitest';
import { Graph } from '@onnx9000/core';
import { CNTKParser, CNTKNode } from '../../src/mmdnn/cntk/parser.js';
import { CNTKMapper } from '../../src/mmdnn/cntk/mapper.js';

describe('CNTK Parser', () => {
  it('should parse CNTK Dictionary into CNTKModel with all fields', () => {
    const parser = new CNTKParser();
    const mockDict = {
      inputs: ['input1'],
      outputs: ['output1'],
      nodes: [
        {
          name: 'conv1',
          uid: 'conv1_uid',
          op: 'Convolution',
          inputs: ['input1', 'weight1'],
          attributes: {
            strides: [1, 1],
            pads: [1, 1, 1, 1],
          },
        },
      ],
    };

    const model = parser.parse(mockDict);
    expect(model.inputs).toEqual(['input1']);
    expect(model.outputs).toEqual(['output1']);
    expect(model.nodes.length).toBe(1);
    expect(model.nodes[0].name).toBe('conv1');
    expect(model.nodes[0].uid).toBe('conv1_uid');
    expect(model.nodes[0].op).toBe('Convolution');
    expect(model.nodes[0].inputs).toEqual(['input1', 'weight1']);
    expect(model.nodes[0].attributes).toEqual({
      strides: [1, 1],
      pads: [1, 1, 1, 1],
    });
  });

  it('should parse CNTK Dictionary with missing fields', () => {
    const parser = new CNTKParser();
    const mockDict = {
      nodes: [
        {
          // missing name, op, inputs, attributes, uid
        },
        {
          uid: 'only_uid',
        },
        {
          name: 'only_name',
        },
      ],
    };

    const model = parser.parse(mockDict);
    expect(model.inputs).toEqual([]);
    expect(model.outputs).toEqual([]);
    expect(model.nodes.length).toBe(3);

    expect(model.nodes[0].name).toBe('unknown');
    expect(model.nodes[0].uid).toBe('unknown');
    expect(model.nodes[0].op).toBe('unknown');
    expect(model.nodes[0].inputs).toEqual([]);
    expect(model.nodes[0].attributes).toEqual({});

    expect(model.nodes[1].name).toBe('only_uid');
    expect(model.nodes[1].uid).toBe('only_uid');

    expect(model.nodes[2].name).toBe('only_name');
    expect(model.nodes[2].uid).toBe('only_name');
  });

  it('should handle completely empty dictionary', () => {
    const parser = new CNTKParser();
    const model = parser.parse({});
    expect(model.inputs).toEqual([]);
    expect(model.outputs).toEqual([]);
    expect(model.nodes).toEqual([]);
  });
});

describe('CNTK Mapper', () => {
  const mapper = new CNTKMapper();
  const graph = new Graph('test');

  const createNode = (
    op: string,
    attributes: Record<string, any> = {},
    inputs: string[] = ['in1', 'in2'],
  ): CNTKNode => ({
    name: `test_${op}`,
    uid: `uid_${op}`,
    op,
    inputs,
    attributes,
  });

  it('maps Convolution with all attributes', () => {
    const layer = createNode('Convolution', {
      strides: [2, 2],
      pads: [1, 1, 1, 1],
      dilations: [2, 2],
      group: 2,
    });
    const nodes = mapper.map(layer, graph);
    expect(nodes.length).toBe(1);
    const node = nodes[0];
    expect(node.opType).toBe('Conv');
    expect(node.inputs).toEqual(['in1', 'in2']);
    expect(node.outputs).toEqual(['uid_Convolution']);
    expect(node.attributes['strides'].value).toEqual([2, 2]);
    expect(node.attributes['pads'].value).toEqual([1, 1, 1, 1]);
    expect(node.attributes['dilations'].value).toEqual([2, 2]);
    expect(node.attributes['group'].value).toEqual(2);
  });

  it('maps Convolution with no attributes', () => {
    const layer = createNode('Convolution', {});
    const nodes = mapper.map(layer, graph);
    expect(nodes.length).toBe(1);
    const node = nodes[0];
    expect(node.opType).toBe('Conv');
    expect(node.attributes['strides']).toBeUndefined();
    expect(node.attributes['pads']).toBeUndefined();
    expect(node.attributes['dilations']).toBeUndefined();
    expect(node.attributes['group']).toBeUndefined();
  });

  it('maps Plus', () => {
    const layer = createNode('Plus');
    const nodes = mapper.map(layer, graph);
    expect(nodes[0].opType).toBe('Add');
  });

  it('maps Minus', () => {
    const layer = createNode('Minus');
    const nodes = mapper.map(layer, graph);
    expect(nodes[0].opType).toBe('Sub');
  });

  it('maps ElementTimes', () => {
    const layer = createNode('ElementTimes');
    const nodes = mapper.map(layer, graph);
    expect(nodes[0].opType).toBe('Mul');
  });

  it('maps Times', () => {
    const layer = createNode('Times');
    const nodes = mapper.map(layer, graph);
    expect(nodes[0].opType).toBe('MatMul');
  });

  it('maps RectifiedLinear', () => {
    const layer = createNode('RectifiedLinear');
    const nodes = mapper.map(layer, graph);
    expect(nodes[0].opType).toBe('Relu');
  });

  it('maps Sigmoid', () => {
    const layer = createNode('Sigmoid');
    const nodes = mapper.map(layer, graph);
    expect(nodes[0].opType).toBe('Sigmoid');
  });

  it('maps Tanh', () => {
    const layer = createNode('Tanh');
    const nodes = mapper.map(layer, graph);
    expect(nodes[0].opType).toBe('Tanh');
  });

  it('maps Softmax with and without axis', () => {
    let layer = createNode('Softmax', { axis: 1 });
    let nodes = mapper.map(layer, graph);
    expect(nodes[0].opType).toBe('Softmax');
    expect(nodes[0].attributes['axis'].value).toBe(1);

    layer = createNode('Softmax', {});
    nodes = mapper.map(layer, graph);
    expect(nodes[0].opType).toBe('Softmax');
    expect(nodes[0].attributes['axis']).toBeUndefined();
  });

  it('maps Pooling with attributes', () => {
    let layer = createNode('Pooling', {
      poolingType: 'Average',
      kernel_shape: [3, 3],
      strides: [2, 2],
      pads: [1, 1, 1, 1],
    });
    let nodes = mapper.map(layer, graph);
    expect(nodes[0].opType).toBe('AveragePool');
    expect(nodes[0].attributes['kernel_shape'].value).toEqual([3, 3]);
    expect(nodes[0].attributes['strides'].value).toEqual([2, 2]);
    expect(nodes[0].attributes['pads'].value).toEqual([1, 1, 1, 1]);

    const maxLayer = createNode('Pooling', { poolingType: 'Max' });
    const maxNodes = mapper.map(maxLayer, graph);
    expect(maxNodes[0].opType).toBe('MaxPool');
    expect(maxNodes[0].attributes['kernel_shape']).toBeUndefined();
    expect(maxNodes[0].attributes['strides']).toBeUndefined();
    expect(maxNodes[0].attributes['pads']).toBeUndefined();
  });

  it('maps AveragePooling explicitly', () => {
    let layer = createNode('AveragePooling', {
      kernel_shape: [3, 3],
      strides: [2, 2],
      pads: [1, 1, 1, 1],
      count_include_pad: 1,
    });
    let nodes = mapper.map(layer, graph);
    expect(nodes[0].opType).toBe('AveragePool');
    expect(nodes[0].attributes['kernel_shape'].value).toEqual([3, 3]);
    expect(nodes[0].attributes['strides'].value).toEqual([2, 2]);
    expect(nodes[0].attributes['pads'].value).toEqual([1, 1, 1, 1]);
    expect(nodes[0].attributes['count_include_pad'].value).toEqual(1);

    layer = createNode('AveragePooling', {});
    nodes = mapper.map(layer, graph);
    expect(nodes[0].opType).toBe('AveragePool');
    expect(nodes[0].attributes['kernel_shape']).toBeUndefined();
    expect(nodes[0].attributes['strides']).toBeUndefined();
    expect(nodes[0].attributes['pads']).toBeUndefined();
    expect(nodes[0].attributes['count_include_pad'].value).toEqual(0); // default
  });

  it('maps BatchNormalization with and without attributes', () => {
    let layer = createNode('BatchNormalization', { epsilon: 1e-5, momentum: 0.9 });
    let nodes = mapper.map(layer, graph);
    expect(nodes[0].opType).toBe('BatchNormalization');
    expect(nodes[0].attributes['epsilon'].value).toBe(1e-5);
    expect(nodes[0].attributes['momentum'].value).toBe(0.9);

    layer = createNode('BatchNormalization', {});
    nodes = mapper.map(layer, graph);
    expect(nodes[0].opType).toBe('BatchNormalization');
    expect(nodes[0].attributes['epsilon']).toBeUndefined();
    expect(nodes[0].attributes['momentum']).toBeUndefined();
  });

  it('maps Splice with and without axis', () => {
    let layer = createNode('Splice', { axis: 0 });
    let nodes = mapper.map(layer, graph);
    expect(nodes[0].opType).toBe('Concat');
    expect(nodes[0].attributes['axis'].value).toBe(0);

    layer = createNode('Splice', {});
    nodes = mapper.map(layer, graph);
    expect(nodes[0].opType).toBe('Concat');
    expect(nodes[0].attributes['axis'].value).toBe(-1); // default
  });

  it('maps Reshape', () => {
    const layer = createNode('Reshape');
    const nodes = mapper.map(layer, graph);
    expect(nodes[0].opType).toBe('Reshape');
  });

  it('maps Transpose with and without perm', () => {
    let layer = createNode('Transpose', { perm: [0, 2, 3, 1] });
    let nodes = mapper.map(layer, graph);
    expect(nodes[0].opType).toBe('Transpose');
    expect(nodes[0].attributes['perm'].value).toEqual([0, 2, 3, 1]);

    layer = createNode('Transpose', {});
    nodes = mapper.map(layer, graph);
    expect(nodes[0].opType).toBe('Transpose');
    expect(nodes[0].attributes['perm']).toBeUndefined();
  });

  it('handles unknown OP gracefully', () => {
    const layer = createNode('UnknownOp');
    const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const nodes = mapper.map(layer, graph);
    expect(nodes).toEqual([]);
    expect(consoleSpy).toHaveBeenCalledWith('Unsupported CNTK node type: UnknownOp');
    consoleSpy.mockRestore();
  });
});
