/* eslint-disable */
// @ts-nocheck
import { Graph, Node, Tensor, Attribute, ValueInfo, Shape, DType } from '@onnx9000/core';
import { NcnnNode, NcnnParam, NcnnBinParser } from './parser.js';

function getInt(attrs: Record<string, string>, key: string, def: number): number {
  return attrs[key] !== undefined ? parseInt(attrs[key], 10) : def;
}

function getFloat(attrs: Record<string, string>, key: string, def: number): number {
  return attrs[key] !== undefined ? parseFloat(attrs[key]) : def;
}

export class NcnnMapper {
  private graph: Graph;
  private bin: NcnnBinParser;
  private tensors = new Map<string, Tensor>();

  constructor(param: NcnnParam, binBuffer: ArrayBuffer) {
    this.graph = new Graph('ncnn_mapped');
    this.bin = new NcnnBinParser(binBuffer);
    this.mapNodes(param.nodes);
  }

  getGraph(): Graph {
    return this.graph;
  }

  private mapNodes(nodes: NcnnNode[]) {
    for (const node of nodes) {
      switch (node.type) {
        case 'Input':
          this.mapInput(node);
          break;
        case 'Convolution':
          this.mapConvolution(node);
          break;
        case 'Pooling':
          this.mapPooling(node);
          break;
        case 'InnerProduct':
          this.mapInnerProduct(node);
          break;
        case 'ReLU':
          this.mapReLU(node);
          break;
        case 'Eltwise':
          this.mapEltwise(node);
          break;
        case 'Concat':
          this.mapConcat(node);
          break;
        case 'Split':
          this.mapSplit(node);
          break;
        case 'Quantize':
          this.mapQuantize(node);
          break;
        case 'Dequantize':
          this.mapDequantize(node);
          break;
        default:
          // Create an identity or generic node for unknown ops
          this.mapGeneric(node);
          break;
      }
    }
  }

  private mapInput(node: NcnnNode) {
    const w = getInt(node.attrs, '0', 0);
    const h = getInt(node.attrs, '1', 0);
    const c = getInt(node.attrs, '2', 0);

    const shape = [-1, c, h, w];
    const vi = new ValueInfo(node.tops[0] || '', shape, 'float32');
    this.graph.inputs.push(vi);
  }

  private mapConvolution(node: NcnnNode) {
    const numOutput = getInt(node.attrs, '0', 0);
    const kernelW = getInt(node.attrs, '1', 0);
    const kernelH = getInt(node.attrs, '11', kernelW);
    const dilationW = getInt(node.attrs, '2', 1);
    const dilationH = getInt(node.attrs, '12', dilationW);
    const strideW = getInt(node.attrs, '3', 1);
    const strideH = getInt(node.attrs, '13', strideW);
    const padW = getInt(node.attrs, '4', 0);
    const padH = getInt(node.attrs, '14', padW);
    const biasTerm = getInt(node.attrs, '5', 0);
    const weightDataSize = getInt(node.attrs, '6', 0);
    const int8ScaleTerm = getInt(node.attrs, '9', 0); // specific INT8 indicator

    const onnxNode = new Node('Conv', [], [], {}, node.name);
    const b0 = node.bottoms[0];
    if (b0) onnxNode.inputs.push(b0);
    const t0 = node.tops[0];
    if (t0) onnxNode.outputs.push(t0);

    onnxNode.attributes['kernel_shape'] = new Attribute('kernel_shape', 'INTS', [kernelH, kernelW]);
    onnxNode.attributes['strides'] = new Attribute('strides', 'INTS', [strideH, strideW]);
    onnxNode.attributes['dilations'] = new Attribute('dilations', 'INTS', [dilationH, dilationW]);
    onnxNode.attributes['pads'] = new Attribute('pads', 'INTS', [padH, padW, padH, padW]);

    const wName = node.name + '_w';
    const weightFloats = this.bin.readFloats(weightDataSize);
    // Approximation: input_channels = weightDataSize / (numOutput * kernelW * kernelH)
    const inC = weightDataSize / (numOutput * kernelW * kernelH);
    const wTensor = new Tensor(wName, [numOutput, inC, kernelH, kernelW], 'float32', true);
    wTensor.data = weightFloats;
    this.graph.initializers.push(wTensor.name);
    const wName_val = wName;
    if (wName_val) onnxNode.inputs.push(wName_val);

    if (biasTerm === 1) {
      const bName = node.name + '_b';
      const biasFloats = this.bin.readFloats(numOutput);
      const bTensor = new Tensor(bName, [numOutput], 'float32', true);
      bTensor.data = biasFloats;
      this.graph.initializers.push(bTensor.name);
      const bName_val = bName;
      if (bName_val) onnxNode.inputs.push(bName_val);
    }

    // Handle INT8 quantized topologies
    if (int8ScaleTerm) {
      onnxNode.attributes['quantized'] = new Attribute('quantized', 'INT', 1);
    }

    this.graph.nodes.push(onnxNode);
  }

  private mapPooling(node: NcnnNode) {
    const poolType = getInt(node.attrs, '0', 0);
    const kernelW = getInt(node.attrs, '1', 0);
    const kernelH = getInt(node.attrs, '11', kernelW);
    const strideW = getInt(node.attrs, '2', 1);
    const strideH = getInt(node.attrs, '12', strideW);
    const padW = getInt(node.attrs, '3', 0);
    const padH = getInt(node.attrs, '13', padW);
    const globalPool = getInt(node.attrs, '4', 0);

    let op = 'MaxPool';
    if (poolType === 1) op = 'AveragePool';

    if (globalPool === 1) {
      if (poolType === 0) op = 'GlobalMaxPool';
      else op = 'GlobalAveragePool';
    }

    const onnxNode = new Node(op, [], [], {}, node.name);
    const b0 = node.bottoms[0];
    if (b0) onnxNode.inputs.push(b0);
    const t0 = node.tops[0];
    if (t0) onnxNode.outputs.push(t0);

    if (globalPool === 0) {
      onnxNode.attributes['kernel_shape'] = new Attribute('kernel_shape', 'INTS', [
        kernelH,
        kernelW,
      ]);
      onnxNode.attributes['strides'] = new Attribute('strides', 'INTS', [strideH, strideW]);
      onnxNode.attributes['pads'] = new Attribute('pads', 'INTS', [padH, padW, padH, padW]);
    }

    this.graph.nodes.push(onnxNode);
  }

  private mapInnerProduct(node: NcnnNode) {
    const numOutput = getInt(node.attrs, '0', 0);
    const biasTerm = getInt(node.attrs, '1', 0);
    const weightDataSize = getInt(node.attrs, '2', 0);

    const onnxNode = new Node('Gemm', [], [], {}, node.name);
    const b0 = node.bottoms[0];
    if (b0) onnxNode.inputs.push(b0);
    const t0 = node.tops[0];
    if (t0) onnxNode.outputs.push(t0);
    onnxNode.attributes['transB'] = new Attribute('transB', 'INT', 1);

    const wName = node.name + '_w';
    const weightFloats = this.bin.readFloats(weightDataSize);
    const inFeatures = weightDataSize / numOutput;
    const wTensor = new Tensor(wName, [numOutput, inFeatures], 'float32', true);
    wTensor.data = weightFloats;
    this.graph.initializers.push(wTensor.name);
    const wName_val = wName;
    if (wName_val) onnxNode.inputs.push(wName_val);

    if (biasTerm === 1) {
      const bName = node.name + '_b';
      const biasFloats = this.bin.readFloats(numOutput);
      const bTensor = new Tensor(bName, [numOutput], 'float32', true);
      bTensor.data = biasFloats;
      this.graph.initializers.push(bTensor.name);
      const bName_val = bName;
      if (bName_val) onnxNode.inputs.push(bName_val);
    }

    this.graph.nodes.push(onnxNode);
  }

  private mapReLU(node: NcnnNode) {
    const slope = getFloat(node.attrs, '0', 0.0);

    const op = slope > 0 ? 'LeakyRelu' : 'Relu';
    const onnxNode = new Node(op, [], [], {}, node.name);
    const b0 = node.bottoms[0];
    if (b0) onnxNode.inputs.push(b0);
    const t0 = node.tops[0];
    if (t0) onnxNode.outputs.push(t0);

    if (slope > 0) {
      onnxNode.attributes['alpha'] = new Attribute('alpha', 'FLOAT', slope);
    }

    this.graph.nodes.push(onnxNode);
  }

  private mapEltwise(node: NcnnNode) {
    const opType = getInt(node.attrs, '0', 0);
    let op = 'Mul'; // 0 = prod
    if (opType === 1) op = 'Add';
    if (opType === 2) op = 'Max';

    const onnxNode = new Node(op, [], [], {}, node.name);
    for (const bottom of node.bottoms) {
      const bottom_val = bottom;
      if (bottom_val) onnxNode.inputs.push(bottom_val);
    }
    const t0 = node.tops[0];
    if (t0) onnxNode.outputs.push(t0);
    this.graph.nodes.push(onnxNode);
  }

  private mapConcat(node: NcnnNode) {
    const axis = getInt(node.attrs, '0', 1);
    const onnxNode = new Node('Concat', [], [], {}, node.name);
    for (const bottom of node.bottoms) {
      const bottom_val = bottom;
      if (bottom_val) onnxNode.inputs.push(bottom_val);
    }
    const t0 = node.tops[0];
    if (t0) onnxNode.outputs.push(t0);
    onnxNode.attributes['axis'] = new Attribute('axis', 'INT', axis);
    this.graph.nodes.push(onnxNode);
  }

  private mapSplit(node: NcnnNode) {
    // NCNN split routes one bottom to multiple tops
    // In ONNX this can be Identity nodes or just re-using the input
    for (let i = 0; i < node.tops.length; i++) {
      const onnxNode = new Node('Identity', [], [], {}, `${node.name}_split_${i}`);
      const b0 = node.bottoms[0];
      if (b0) onnxNode.inputs.push(b0);
      const ti = node.tops[i];
      if (ti) onnxNode.outputs.push(ti);
      this.graph.nodes.push(onnxNode);
    }
  }

  private mapQuantize(node: NcnnNode) {
    // NCNN specific INT8 mapping back to ONNX QuantizeLinear
    const scale = getFloat(node.attrs, '0', 1.0);
    const onnxNode = new Node('QuantizeLinear', [], [], {}, node.name);
    const b0 = node.bottoms[0];
    if (b0) onnxNode.inputs.push(b0);

    const scaleName = node.name + '_scale';
    const scaleTensor = new Tensor(scaleName, [1], 'float32', true);
    scaleTensor.data = new Float32Array([scale]);
    this.graph.tensors[scaleTensor.name] = scaleTensor;
    this.graph.initializers.push(scaleTensor.name);
    const scaleName_val = scaleName;
    if (scaleName_val) onnxNode.inputs.push(scaleName_val);

    // Zero point is usually 0 for NCNN
    const zpName = node.name + '_zp';
    const zpTensor = new Tensor(zpName, [1], 'int8', true);
    zpTensor.data = new Int8Array([0]);
    this.graph.tensors[zpTensor.name] = zpTensor;
    this.graph.initializers.push(zpTensor.name);
    const zpName_val = zpName;
    if (zpName_val) onnxNode.inputs.push(zpName_val);

    const t0 = node.tops[0];
    if (t0) onnxNode.outputs.push(t0);
    this.graph.nodes.push(onnxNode);
  }

  private mapDequantize(node: NcnnNode) {
    const scale = getFloat(node.attrs, '0', 1.0);
    const onnxNode = new Node('DequantizeLinear', [], [], {}, node.name);
    const b0 = node.bottoms[0];
    if (b0) onnxNode.inputs.push(b0);

    const scaleName = node.name + '_scale';
    const scaleTensor = new Tensor(scaleName, [1], 'float32', true);
    scaleTensor.data = new Float32Array([scale]);
    this.graph.tensors[scaleTensor.name] = scaleTensor;
    this.graph.initializers.push(scaleTensor.name);
    const scaleName_val = scaleName;
    if (scaleName_val) onnxNode.inputs.push(scaleName_val);

    const zpName = node.name + '_zp';
    const zpTensor = new Tensor(zpName, [1], 'int8', true);
    zpTensor.data = new Int8Array([0]);
    this.graph.tensors[zpTensor.name] = zpTensor;
    this.graph.initializers.push(zpTensor.name);
    const zpName_val = zpName;
    if (zpName_val) onnxNode.inputs.push(zpName_val);

    const t0 = node.tops[0];
    if (t0) onnxNode.outputs.push(t0);
    this.graph.nodes.push(onnxNode);
  }

  private mapGeneric(node: NcnnNode) {
    const onnxNode = new Node(node.type, [], [], {}, node.name);
    for (const bottom of node.bottoms) {
      if (bottom) onnxNode.inputs.push(bottom);
    }
    for (const top of node.tops) {
      if (top) onnxNode.outputs.push(top);
    }

    // Just store attrs as string attributes
    for (const [k, v] of Object.entries(node.attrs)) {
      onnxNode.attributes[`ncnn_attr_${k}`] = new Attribute(`ncnn_attr_${k}`, 'STRING', v);
    }

    this.graph.nodes.push(onnxNode);
  }
}
