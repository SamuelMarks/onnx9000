import { describe, it, expect } from 'vitest';
import { CaffeMapper, register_caffe_op } from '../../src/mmdnn/caffe/mapper.js';
import { parsePrototxt, parseCaffemodel } from '../../src/mmdnn/caffe/parser.js';
import { Graph } from '@onnx9000/core';

// Helper to encode varint
function encodeVarInt(value: number): number[] {
  const bytes: number[] = [];
  while (value > 127) {
    bytes.push((value & 127) | 128);
    value >>>= 7;
  }
  bytes.push(value);
  return bytes;
}

function encodeTag(fieldNumber: number, wireType: number): number[] {
  return encodeVarInt((fieldNumber << 3) | wireType);
}

function encodeString(str: string): number[] {
  return Array.from(new TextEncoder().encode(str));
}

function encodeLengthDelimited(fieldNumber: number, bytes: number[]): number[] {
  return [...encodeTag(fieldNumber, 2), ...encodeVarInt(bytes.length), ...bytes];
}

function encodeFloat32(fieldNumber: number, value: number): number[] {
  const buf = new ArrayBuffer(4);
  new DataView(buf).setFloat32(0, value, true);
  return [...encodeTag(fieldNumber, 5), ...Array.from(new Uint8Array(buf))];
}

function encodeVarIntField(fieldNumber: number, value: number): number[] {
  return [...encodeTag(fieldNumber, 0), ...encodeVarInt(value)];
}

describe('MMDNN - Caffe Importer Parser', () => {
  describe('parsePrototxt', () => {
    it('should parse simple prototxt', () => {
      const text = `
name: "TestNet"
input: "data"
input_dim: 1
input_dim: 3
dim: 224
bottom: "a"
top: "b"
layer {
  name: 'conv1'
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  convolution_param {
    num_output: 64
    kernel_size: 7
    stride: 2
    pad: 3
    bias_term: false
    group: true
  }
}
      `;
      const result = parsePrototxt(text);
      expect(result.name).toBe('TestNet');
      expect(result.input).toEqual(['data']);
      expect(result.input_dim).toEqual([1, 3]);
      expect(result.dim).toEqual([224]);
      expect(result.bottom).toEqual(['a']);
      expect(result.top).toEqual(['b']);
      expect(result.layer.length).toBe(1);
      expect(result.layer[0].name).toBe('conv1');
      expect(result.layer[0].convolution_param.bias_term).toBe(false);
      expect(result.layer[0].convolution_param.group).toBe(true);
    });

    it('should handle layers array and multiple sub-objects', () => {
      const text = `
layers {
  name: "L1"
}
layers {
  name: "L2"
  include {
    phase: TEST
  }
  include {
    phase: TRAIN
  }
}
      `;
      const result = parsePrototxt(text);
      expect(result.layer.length).toBe(2);
      expect(result.layer[1].include.length).toBe(2);
      expect(result.layer[1].include[0].phase).toBe('TEST');
    });

    it('should ignore empty lines and invalid lines', () => {
      const text = `
      # comment
      
      badline
      `;
      const result = parsePrototxt(text);
      expect(result.layer).toEqual([]);
    });

    it('should parse numbers and quotes correctly', () => {
      const text = `
      val1: "string"
      val2: 'string'
      val3: 123.45
      dim {
        dummy: 1
      }
      dim: 2
      `;
      const result = parsePrototxt(text);
      expect(result.val1).toBe('string');
      expect(result.val2).toBe('string');
      expect(result.val3).toBe(123.45);
      expect(Array.isArray(result.dim)).toBe(true);
      expect(result.dim.length).toBe(2);
    });
  });

  describe('parseCaffemodel', () => {
    it('should parse binary caffe model', async () => {
      // Create a BlobShape (7)
      const shapeBytes = [
        ...encodeVarIntField(1, 1),
        ...encodeLengthDelimited(1, [...encodeVarInt(2), ...encodeVarInt(3)]),
      ];
      // Add unknown field to BlobShape to hit skipField
      shapeBytes.push(...encodeVarIntField(99, 1));

      // Create a BlobProto
      const blobBytes = [
        ...encodeVarIntField(1, 1), // num
        ...encodeVarIntField(2, 3), // channels
        ...encodeVarIntField(3, 224), // height
        ...encodeVarIntField(4, 224), // width
        ...encodeFloat32(5, 1.5), // data (32BIT)
        ...encodeLengthDelimited(5, Array.from(new Uint8Array(new Float32Array([2.5]).buffer))), // data (LENGTH_DELIMITED)
        ...encodeVarIntField(5, 99), // data (UNKNOWN WIRE TYPE TO TRIGGER SKIP)
        ...encodeLengthDelimited(7, shapeBytes), // shape
        ...encodeVarIntField(99, 1), // skipField
      ];

      // Create a BlobProto without shape to trigger fallback
      const blobNoShapeBytes = [...encodeVarIntField(1, 10), ...encodeVarIntField(2, 20)];

      // Create LayerParameter (100)
      const layerBytes = [
        ...encodeLengthDelimited(1, encodeString('conv1')), // name
        ...encodeLengthDelimited(2, encodeString('Convolution')), // type
        ...encodeLengthDelimited(50, blobBytes), // blobs
        ...encodeLengthDelimited(6, blobNoShapeBytes), // blobs V1
        ...encodeVarIntField(99, 1), // skipField
      ];

      // Create LayerParameter V1 (2)
      const layerV1Bytes = [...encodeLengthDelimited(1, encodeString('conv2'))];

      // Create NetParameter
      const netBytes = [
        ...encodeLengthDelimited(1, encodeString('MyNet')), // name
        ...encodeLengthDelimited(100, layerBytes), // layer
        ...encodeLengthDelimited(2, layerV1Bytes), // layer V1
        ...encodeVarIntField(99, 1), // skipField
      ];

      const result = await parseCaffemodel(new Uint8Array(netBytes));
      expect(result.name).toBe('MyNet');
      expect(result.layer.length).toBe(2);
      expect(result.layer[0].name).toBe('conv1');
      expect(result.layer[0].type).toBe('Convolution');
      expect(result.layer[0].blobs.length).toBe(2);

      const blob = result.layer[0].blobs[0];
      expect(blob.data[0]).toBeCloseTo(1.5);
      expect(blob.data[1]).toBeCloseTo(2.5);
      expect(blob.shape).toEqual([1, 2, 3]);

      const blob2 = result.layer[0].blobs[1];
      expect(blob2.shape).toEqual([10, 20]);
    });
  });
});

describe('MMDNN - Caffe Importer Mapping', () => {
  const mapper = new CaffeMapper();
  const graph = new Graph('caffe_test');

  it('should map unknown layer to empty array', () => {
    const layer = { type: 'Unknown', name: 'u' };
    const nodes = mapper.map(layer, graph);
    expect(nodes[0].opType).toEqual('Identity');
  });

  it('should resolve paddings', () => {
    // using Convolution to test resolvePadding
    const nodes1 = mapper.map(
      {
        type: 'Convolution',
        name: 'c',
        convolution_param: { pad: [2] },
      },
      graph,
    );
    expect(nodes1[0].attributes['pads'].value).toEqual([2, 2, 2, 2]);

    const nodes2 = mapper.map(
      {
        type: 'Convolution',
        name: 'c',
        convolution_param: { pad_h: 2, pad_w: 3 },
      },
      graph,
    );
    expect(nodes2[0].attributes['pads'].value).toEqual([2, 3, 2, 3]);

    const nodes3 = mapper.map(
      {
        type: 'Convolution',
        name: 'c',
      },
      graph,
    );
    expect(nodes3[0].attributes['pads'].value).toEqual([0, 0, 0, 0]);
  });

  it('should map Convolution', () => {
    const layer = {
      name: 'conv1',
      type: 'Convolution',
      bottom: ['data'],
      top: ['conv1'],
      convolution_param: {
        pad: 1,
        kernel_size: 3,
        stride: 2,
        group: 2,
        dilation: 2,
      },
      blobs: [1, 2],
    };

    const nodes = mapper.map(layer, graph);
    expect(nodes.length).toBe(1);
    const node = nodes[0];
    expect(node.opType).toBe('Conv');
    expect(node.inputs).toEqual(['data', 'conv1_W', 'conv1_B']);
    expect(node.outputs).toEqual(['conv1']);
    expect(node.attributes['pads'].value).toEqual([1, 1, 1, 1]);
    expect(node.attributes['kernel_shape'].value).toEqual([3, 3]);
    expect(node.attributes['strides'].value).toEqual([2, 2]);
    expect(node.attributes['group'].value).toBe(2);
    expect(node.attributes['dilations'].value).toEqual([2, 2]);
  });

  it('should map Convolution with arrays', () => {
    const layer = {
      name: 'conv1',
      type: 'Convolution',
      convolution_param: {
        kernel_size: [3],
        stride: [2],
        dilation: [2],
      },
    };
    const nodes = mapper.map(layer, graph);
    expect(nodes[0].attributes['kernel_shape'].value).toEqual([3, 3]);
    expect(nodes[0].attributes['strides'].value).toEqual([2, 2]);
    expect(nodes[0].attributes['dilations'].value).toEqual([2, 2]);
  });

  it('should map Convolution with h/w', () => {
    const layer = {
      name: 'conv1',
      type: 'Convolution',
      convolution_param: {
        kernel_h: 3,
        kernel_w: 4,
        stride_h: 1,
        stride_w: 2,
      },
    };
    const nodes = mapper.map(layer, graph);
    expect(nodes[0].attributes['kernel_shape'].value).toEqual([3, 4]);
    expect(nodes[0].attributes['strides'].value).toEqual([1, 2]);
  });

  it('should map InnerProduct', () => {
    const layer = {
      name: 'fc1',
      type: 'InnerProduct',
      bottom: ['conv1'],
      blobs: [1], // only weights
    };

    const nodes = mapper.map(layer, graph);
    expect(nodes[0].opType).toBe('Gemm');
    expect(nodes[0].inputs).toEqual(['conv1', 'fc1_W']);

    const layerWithBias = {
      name: 'fc2',
      type: 'InnerProduct',
      bottom: ['conv2'],
      blobs: [1, 2], // weights and bias
    };
    const nodesWithBias = mapper.map(layerWithBias, graph);
    expect(nodesWithBias[0].inputs).toEqual(['conv2', 'fc2_W', 'fc2_B']);
  });

  it('should map ReLU', () => {
    const nodes = mapper.map({ name: 'relu1', type: 'ReLU' }, graph);
    expect(nodes[0].opType).toBe('Relu');

    const nodesLeaky = mapper.map(
      {
        name: 'relu2',
        type: 'ReLU',
        relu_param: { negative_slope: 0.1 },
      },
      graph,
    );
    expect(nodesLeaky[0].opType).toBe('LeakyRelu');
  });

  it('should map Pooling', () => {
    const nodesMax = mapper.map(
      {
        name: 'pool1',
        type: 'Pooling',
        pooling_param: { pool: 0 },
      },
      graph,
    );
    expect(nodesMax[0].opType).toBe('MaxPool');

    const nodesAve = mapper.map(
      {
        name: 'pool2',
        type: 'Pooling',
        pooling_param: { pool: 1 },
      },
      graph,
    );
    expect(nodesAve[0].opType).toBe('AveragePool');
  });

  it('should map LRN', () => {
    const nodes = mapper.map(
      {
        name: 'norm1',
        type: 'LRN',
        lrn_param: {},
      },
      graph,
    );
    expect(nodes[0].opType).toBe('LRN');
    expect(nodes[0].attributes['size'].value).toBe(5);
    expect(nodes[0].attributes['alpha'].value).toBe(1.0);
    expect(nodes[0].attributes['beta'].value).toBe(0.75);
    expect(nodes[0].attributes['bias'].value).toBe(1.0);
  });

  it('should map Softmax', () => {
    const nodes = mapper.map(
      {
        name: 'sm',
        type: 'Softmax',
      },
      graph,
    );
    expect(nodes[0].opType).toBe('Softmax');
    expect(nodes[0].attributes['axis'].value).toBe(1);
  });

  it('should map Eltwise', () => {
    expect(mapper.map({ type: 'Eltwise', eltwise_param: { operation: 0 } }, graph)[0].opType).toBe(
      'Mul',
    );
    expect(mapper.map({ type: 'Eltwise', eltwise_param: { operation: 1 } }, graph)[0].opType).toBe(
      'Add',
    );
    expect(mapper.map({ type: 'Eltwise', eltwise_param: { operation: 2 } }, graph)[0].opType).toBe(
      'Max',
    );
    expect(mapper.map({ type: 'Eltwise' }, graph)[0].opType).toBe('Add');
  });

  it('should map Concat', () => {
    const nodes = mapper.map({ type: 'Concat' }, graph);
    expect(nodes[0].opType).toBe('Concat');
    expect(nodes[0].attributes['axis'].value).toBe(1);
  });

  it('should map Scale', () => {
    const layer = {
      name: 'scale1',
      type: 'Scale',
      blobs: [1, 2],
    };
    const nodes = mapper.map(layer, graph);
    expect(nodes.length).toBe(2);
    expect(nodes[0].opType).toBe('Mul');
    expect(nodes[1].opType).toBe('Add');

    const layerNoBias = {
      name: 'scale2',
      type: 'Scale',
      blobs: [1],
    };
    const nodesNoBias = mapper.map(layerNoBias, graph);
    expect(nodesNoBias.length).toBe(1);
    expect(nodesNoBias[0].opType).toBe('Mul');

    const layerNoBlobs = {
      name: 'scale3',
      type: 'Scale',
    };
    const nodesNoBlobs = mapper.map(layerNoBlobs, graph);
    expect(nodesNoBlobs.length).toBe(1);
    expect(nodesNoBlobs[0].inputs).toContain('scale3_scale_in');
  });

  it('should map BatchNorm', () => {
    const nodes = mapper.map({ type: 'BatchNorm', name: 'bn' }, graph);
    expect(nodes[0].opType).toBe('BatchNormalization');
    expect(nodes[0].inputs).toContain('bn_scale');
  });

  it('should map Dropout', () => {
    const nodes = mapper.map({ type: 'Dropout' }, graph);
    expect(nodes[0].opType).toBe('Dropout');
    expect(nodes[0].attributes['ratio'].value).toBe(0.5);
  });

  it('should map Reshape', () => {
    const nodes = mapper.map({ type: 'Reshape', name: 'rs' }, graph);
    expect(nodes[0].opType).toBe('Reshape');
    expect(nodes[0].inputs).toContain('rs_shape');
  });

  it('should map Flatten', () => {
    const nodes = mapper.map({ type: 'Flatten' }, graph);
    expect(nodes[0].opType).toBe('Flatten');
    expect(nodes[0].attributes['axis'].value).toBe(1);
  });

  it('should map Split', () => {
    const nodes = mapper.map({ type: 'Split', top: ['a', 'b'] }, graph);
    expect(nodes.length).toBe(2);
    expect(nodes[0].opType).toBe('Identity');

    const nodesNoTop = mapper.map({ type: 'Split', name: 's' }, graph);
    expect(nodesNoTop.length).toBe(1);
    expect(nodesNoTop[0].outputs).toEqual(['s']);
  });

  it('should map Slice', () => {
    const nodes = mapper.map(
      {
        type: 'Slice',
        slice_param: { slice_point: [10, 20] },
      },
      graph,
    );
    expect(nodes[0].opType).toBe('Split');
    expect(nodes[0].attributes['split'].value).toEqual([10, 20]);

    const nodesSinglePoint = mapper.map(
      {
        type: 'Slice',
        slice_param: { slice_point: 10 },
      },
      graph,
    );
    expect(nodesSinglePoint[0].attributes['split'].value).toEqual([10]);
  });

  it('should handle decorator properly for different domains', () => {
    class Dummy {
      @register_caffe_op('not_caffe', 'ShouldNotRegister')
      method() {}
    }
    const nodes = mapper.map({ type: 'ShouldNotRegister' }, graph);
    expect(nodes[0].opType).toEqual('Identity');
  });

  it('MMDNN - Caffe Importer Mapping Softmax Default Axis', () => {
    const mapper = new CaffeMapper();
    const graph = new Graph();
    const layer = {
      name: 'soft',
      type: 'Softmax',
      bottom: ['data'],
      top: ['soft'],
    };
    const nodes = mapper.map(layer, graph);
    expect(nodes[0].attributes['axis'].value).toBe(1);
  });
});
