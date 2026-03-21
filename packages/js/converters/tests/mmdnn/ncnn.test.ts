import { describe, it, expect } from 'vitest';
import { parseNcnnParam, NcnnMapper } from '../../src/mmdnn/ncnn/index.js';
import { NcnnBinParser } from '../../src/mmdnn/ncnn/parser.js';

describe('NCNN Parser', () => {
  it('should parse basic NCNN .param file', () => {
    const paramText = `7767517
5 5
Input            data             0 1 data 0=224 1=224 2=3
Convolution      conv1            1 1 data conv1 0=64 1=3 11=3 2=1 3=1 4=1 5=1 6=1728
ReLU             relu1            1 1 conv1 conv1_relu 0=0.000000
Pooling          pool1            1 1 conv1_relu pool1 0=0 1=3 2=2 3=0 4=0
Split            split1           1 2 pool1 pool1_1 pool1_2
`;
    const param = parseNcnnParam(paramText);
    expect(param.magic).toBe(7767517);
    expect(param.layerCount).toBe(5);
    expect(param.blobCount).toBe(5);
    expect(param.nodes.length).toBe(5);

    const conv = param.nodes.find((n) => n.name === 'conv1')!;
    expect(conv.type).toBe('Convolution');
    expect(conv.attrs['0']).toBe('64');
    expect(conv.bottoms).toEqual(['data']);
    expect(conv.tops).toEqual(['conv1']);
  });

  it('should handle file with missing magic and empty lines', () => {
    const paramText = `
# this is a comment

1 1

Input            data             0 1 data 0=224 1=224 2=3
`;
    const param = parseNcnnParam(paramText);
    expect(param.magic).toBe(0);
    expect(param.layerCount).toBe(1);
    expect(param.blobCount).toBe(1);
    expect(param.nodes.length).toBe(1);
  });

  it('should handle malformed attributes including arrays', () => {
    const paramText = `7767517
1 1
CustomOp         custom           1 1 data out 0=224 missing_value= -23309=array_val malformed
`;
    const param = parseNcnnParam(paramText);
    const node = param.nodes[0];
    expect(node.attrs['0']).toBe('224');
    expect(node.attrs['missing_value']).toBe('');
    expect(node.attrs['-23309']).toBe('array_val');
  });

  it('should ignore malformed lines', () => {
    const paramText = `7767517
1 1
ShortLine
Input data 0 1 data 0=224
`;
    const param = parseNcnnParam(paramText);
    expect(param.nodes.length).toBe(1);
    expect(param.nodes[0].name).toBe('data');
  });

  it('should test NcnnBinParser readInts and readBytes', () => {
    const buffer = new ArrayBuffer(16);
    const view = new DataView(buffer);
    view.setInt32(0, 123, true);
    view.setInt32(4, 456, true);
    view.setUint8(8, 10);
    view.setUint8(9, 20);

    const parser = new NcnnBinParser(buffer);
    const ints = parser.readInts(2);
    expect(ints[0]).toBe(123);
    expect(ints[1]).toBe(456);

    const bytes = parser.readBytes(2);
    expect(bytes[0]).toBe(10);
    expect(bytes[1]).toBe(20);
    // align offset should advance it to 12
    const floats = parser.readFloats(1);
    expect(floats.length).toBe(1);
  });
});

describe('NCNN Mapper', () => {
  it('maps Convolution with no bias and no int8', () => {
    const paramText = `7767517
2 2
Input            data             0 1 data 0=224 1=224 2=3
Convolution      conv1            1 1 data conv1 0=64 1=3 2=1 3=1 4=1 5=0 6=1728
`;
    const param = parseNcnnParam(paramText);
    const binBuffer = new ArrayBuffer(1728 * 4);
    const mapper = new NcnnMapper(param, binBuffer);
    const graph = mapper.getGraph();

    const conv = graph.nodes.find((n) => n.opType === 'Conv')!;
    expect(conv.inputs.length).toBe(2); // no bias
    expect(conv.attributes['quantized']).toBeUndefined();
    expect(conv.attributes['kernel_shape']!.value).toEqual([3, 3]);
  });

  it('maps Convolution with bias and int8', () => {
    const paramText = `7767517
2 2
Input            data             0 1 data 0=224 1=224 2=3
Convolution      conv1            1 1 data conv1 0=64 1=3 2=1 3=1 4=1 5=1 6=1728 9=1
`;
    const param = parseNcnnParam(paramText);
    const binBuffer = new ArrayBuffer((1728 + 64) * 4);
    const mapper = new NcnnMapper(param, binBuffer);
    const graph = mapper.getGraph();

    const conv = graph.nodes.find((n) => n.opType === 'Conv')!;
    expect(conv.inputs.length).toBe(3); // data, weight, bias
    expect(conv.attributes['quantized']!.value).toBe(1);
  });

  it('maps Pooling max, average, global', () => {
    const paramText = `7767517
4 4
Input            data             0 1 data 0=224 1=224 2=3
Pooling          pool1            1 1 data pool1 0=0 1=3 2=2 3=0 4=0
Pooling          pool2            1 1 data pool2 0=1 1=3 2=2 3=0 4=0
Pooling          pool3            1 1 data pool3 0=0 1=3 2=2 3=0 4=1
`;
    const param = parseNcnnParam(paramText);
    const mapper = new NcnnMapper(param, new ArrayBuffer(0));
    const graph = mapper.getGraph();

    const maxPool = graph.nodes.find((n) => n.name === 'pool1')!;
    expect(maxPool.opType).toBe('MaxPool');
    expect(maxPool.attributes['kernel_shape']).toBeDefined();

    const avgPool = graph.nodes.find((n) => n.name === 'pool2')!;
    expect(avgPool.opType).toBe('AveragePool');

    const globalPool = graph.nodes.find((n) => n.name === 'pool3')!;
    expect(globalPool.opType).toBe('GlobalMaxPool');
    expect(globalPool.attributes['kernel_shape']).toBeUndefined(); // skipped for global
  });

  it('maps Pooling global average', () => {
    const paramText = `7767517
2 2
Input            data             0 1 data 0=224 1=224 2=3
Pooling          pool1            1 1 data pool1 0=1 4=1
`;
    const param = parseNcnnParam(paramText);
    const mapper = new NcnnMapper(param, new ArrayBuffer(0));
    const graph = mapper.getGraph();

    const globalAvgPool = graph.nodes.find((n) => n.name === 'pool1')!;
    expect(globalAvgPool.opType).toBe('GlobalAveragePool');
  });

  it('maps InnerProduct with and without bias', () => {
    const paramText = `7767517
3 3
Input            data             0 1 data 0=224 1=224 2=3
InnerProduct     ip1              1 1 data ip1 0=100 1=1 2=200
InnerProduct     ip2              1 1 ip1 ip2 0=50 1=0 2=100
`;
    const param = parseNcnnParam(paramText);
    const binBuffer = new ArrayBuffer((200 + 100 + 100) * 4);
    const mapper = new NcnnMapper(param, binBuffer);
    const graph = mapper.getGraph();

    const ip1 = graph.nodes.find((n) => n.name === 'ip1')!;
    expect(ip1.opType).toBe('Gemm');
    expect(ip1.inputs.length).toBe(3); // data, weight, bias
    expect(ip1.attributes['transB']!.value).toBe(1);

    const ip2 = graph.nodes.find((n) => n.name === 'ip2')!;
    expect(ip2.opType).toBe('Gemm');
    expect(ip2.inputs.length).toBe(2); // data, weight
  });

  it('maps ReLU with and without slope', () => {
    const paramText = `7767517
3 3
Input            data             0 1 data 0=224 1=224 2=3
ReLU             relu1            1 1 data relu1 0=0.000000
ReLU             lrelu1           1 1 data lrelu1 0=0.1
`;
    const param = parseNcnnParam(paramText);
    const mapper = new NcnnMapper(param, new ArrayBuffer(0));
    const graph = mapper.getGraph();

    const relu = graph.nodes.find((n) => n.name === 'relu1')!;
    expect(relu.opType).toBe('Relu');
    expect(relu.attributes['alpha']).toBeUndefined();

    const lrelu = graph.nodes.find((n) => n.name === 'lrelu1')!;
    expect(lrelu.opType).toBe('LeakyRelu');
    expect(lrelu.attributes['alpha']!.value).toBeCloseTo(0.1);
  });

  it('maps Eltwise', () => {
    const paramText = `7767517
4 4
Input            data             0 1 data 0=224 1=224 2=3
Eltwise          elt_prod         2 1 data data elt_prod 0=0
Eltwise          elt_add          2 1 data data elt_add 0=1
Eltwise          elt_max          2 1 data data elt_max 0=2
`;
    const param = parseNcnnParam(paramText);
    const mapper = new NcnnMapper(param, new ArrayBuffer(0));
    const graph = mapper.getGraph();

    expect(graph.nodes.find((n) => n.name === 'elt_prod')!.opType).toBe('Mul');
    expect(graph.nodes.find((n) => n.name === 'elt_add')!.opType).toBe('Add');
    expect(graph.nodes.find((n) => n.name === 'elt_max')!.opType).toBe('Max');
  });

  it('maps Concat', () => {
    const paramText = `7767517
2 2
Input            data             0 1 data 0=224 1=224 2=3
Concat           concat1          2 1 data data concat1 0=1
`;
    const param = parseNcnnParam(paramText);
    const mapper = new NcnnMapper(param, new ArrayBuffer(0));
    const graph = mapper.getGraph();

    const concat = graph.nodes.find((n) => n.name === 'concat1')!;
    expect(concat.opType).toBe('Concat');
    expect(concat.attributes['axis']!.value).toBe(1);
    expect(concat.inputs).toEqual(['data', 'data']);
  });

  it('maps Split', () => {
    const paramText = `7767517
2 3
Input            data             0 1 data 0=224 1=224 2=3
Split            split1           1 2 data split_out1 split_out2
`;
    const param = parseNcnnParam(paramText);
    const mapper = new NcnnMapper(param, new ArrayBuffer(0));
    const graph = mapper.getGraph();

    const identities = graph.nodes.filter((n) => n.opType === 'Identity');
    expect(identities.length).toBe(2);
    expect(identities[0].outputs[0]).toBe('split_out1');
    expect(identities[1].outputs[0]).toBe('split_out2');
  });

  it('maps Quantize', () => {
    const paramText = `7767517
2 2
Input            data             0 1 data 0=224 1=224 2=3
Quantize         quant1           1 1 data quant1 0=0.00392157
`;
    const param = parseNcnnParam(paramText);
    const mapper = new NcnnMapper(param, new ArrayBuffer(0));
    const graph = mapper.getGraph();

    const quant = graph.nodes.find((n) => n.opType === 'QuantizeLinear')!;
    expect(quant).toBeDefined();
    expect(quant.inputs[0]).toBe('data');
    expect(quant.outputs[0]).toBe('quant1');
    expect(graph.initializers).toContain('quant1_scale');
    expect(graph.initializers).toContain('quant1_zp');
  });

  it('maps Dequantize', () => {
    const paramText = `7767517
2 2
Input            data             0 1 data 0=224 1=224 2=3
Dequantize       dequant1         1 1 data dequant1 0=0.00392157
`;
    const param = parseNcnnParam(paramText);
    const mapper = new NcnnMapper(param, new ArrayBuffer(0));
    const graph = mapper.getGraph();

    const dequant = graph.nodes.find((n) => n.opType === 'DequantizeLinear')!;
    expect(dequant).toBeDefined();
    expect(dequant.inputs[0]).toBe('data');
    expect(dequant.outputs[0]).toBe('dequant1');
    expect(graph.initializers).toContain('dequant1_scale');
    expect(graph.initializers).toContain('dequant1_zp');
  });

  it('maps unknown Generic ops', () => {
    const paramText = `7767517
2 2
Input            data             0 1 data 0=224 1=224 2=3
UnknownOp        unknown1         1 1 data unknown1 0=1 1=2
`;
    const param = parseNcnnParam(paramText);
    const mapper = new NcnnMapper(param, new ArrayBuffer(0));
    const graph = mapper.getGraph();

    const unknown = graph.nodes.find((n) => n.name === 'unknown1')!;
    expect(unknown.opType).toBe('UnknownOp');
    expect(unknown.attributes['ncnn_attr_0']!.value).toBe('1');
    expect(unknown.attributes['ncnn_attr_1']!.value).toBe('2');
    expect(unknown.inputs).toEqual(['data']);
    expect(unknown.outputs).toEqual(['unknown1']);
  });
});
