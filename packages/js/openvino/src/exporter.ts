import { Graph, Node, Tensor, ValueInfo, DType } from '@onnx9000/core';
import { XmlBuilder, XmlNode } from './xml_builder';

export interface OpenVinoExportOptions {
  version?: string;
  compressToFp16?: boolean;
  clampDynamic?: boolean;
}

export class OpenVinoExporter {
  graph: Graph;
  version: string;
  compressToFp16: boolean;
  clampDynamic: boolean;
  private nodeIdCounter: number = 0;
  layerIds: Map<string, number> = new Map();
  portIds: Map<string, { layerId: number; portId: number }> = new Map();
  binBuffer: Uint8Array[] = [];
  private binCache: Map<string, { offset: number; size: number }> = new Map();
  private edges: XmlNode[] = [];
  private portCounters: Map<number, number> = new Map();

  constructor(graph: Graph, options: OpenVinoExportOptions = {}) {
    this.graph = graph;
    this.version = options.version || '11';
    this.compressToFp16 = options.compressToFp16 || false;
    this.clampDynamic = options.clampDynamic || false;
  }

  private nextId(): number {
    return this.nodeIdCounter++;
  }

  private nextPort(layerId: number): number {
    const p = this.portCounters.get(layerId) || 0;
    this.portCounters.set(layerId, p + 1);
    return p;
  }

  private emitDynamicConst(
    name: string,
    data: number[],
    shape: number[],
    dtype: DType,
  ): { layerNode: XmlNode; portId: number } {
    const layerId = this.nextId();
    this.layerIds.set(name, layerId);

    const layer = new XmlNode('layer');
    layer.setAttribute('id', layerId.toString());
    layer.setAttribute('name', name);
    layer.setAttribute('type', 'Const');
    layer.setAttribute('version', 'opset1');

    const dataNode = new XmlNode('data');
    dataNode.setAttribute('element_type', this.mapDtype(dtype));
    const actualShape = shape.length > 0 ? shape : [1];
    dataNode.setAttribute('shape', actualShape.join(','));

    let byteLength = 0;
    if (dtype === 'int64') byteLength = data.length * 8;
    else if (dtype === 'int32') byteLength = data.length * 4;
    else if (dtype === 'float32') byteLength = data.length * 4;
    else throw new Error(`Dynamic const for ${dtype} not implemented`);

    const buffer = new ArrayBuffer(byteLength);
    const view = new DataView(buffer);
    for (let i = 0; i < data.length; i++) {
      if (dtype === 'int64') {
        view.setBigInt64(i * 8, BigInt(data[i] as number), true); // little-endian
      } else if (dtype === 'int32') {
        view.setInt32(i * 4, data[i] as number, true);
      } else if (dtype === 'float32') {
        view.setFloat32(i * 4, data[i] as number, true);
      }
    }

    const uint8View = new Uint8Array(buffer);
    const hashKey = this.uint8ArrayToString(uint8View);
    const cacheHit = this.binCache.get(hashKey);

    if (cacheHit) {
      dataNode.setAttribute('offset', cacheHit.offset.toString());
      dataNode.setAttribute('size', cacheHit.size.toString());
    } else {
      const totalLength = this.binBuffer.reduce((acc, val) => acc + val.length, 0);
      this.binBuffer.push(uint8View);
      dataNode.setAttribute('offset', totalLength.toString());
      dataNode.setAttribute('size', uint8View.length.toString());
      this.binCache.set(hashKey, { offset: totalLength, size: uint8View.length });
    }

    layer.addChild(dataNode);

    const outputPort = this.nextPort(layerId);
    const outNode = new XmlNode('output');
    const port = this.emitShape(actualShape, 'port');
    port.setAttribute('id', outputPort.toString());
    port.setAttribute('precision', this.mapDtype(dtype));
    outNode.addChild(port);
    layer.addChild(outNode);

    this.portIds.set(name, { layerId, portId: outputPort });
    return { layerNode: layer, portId: outputPort };
  }

  private mapDtype(dtype: DType): string {
    switch (dtype) {
      case 'float32':
        return this.compressToFp16 ? 'f16' : 'f32';
      case 'float64':
        return 'f64';
      case 'float16':
        return 'f16';
      case 'bfloat16':
        return 'bf16';
      case 'int64':
        return 'i64';
      case 'int32':
        return 'i32';
      case 'int16':
        return 'i16';
      case 'int8':
        return 'i8';
      case 'uint64':
        return 'u64';
      case 'uint32':
        return 'u32';
      case 'uint16':
        return 'u16';
      case 'uint8':
        return 'u8';
      case 'bool':
        return 'boolean';
      default:
        throw new Error(`Unsupported dtype for OpenVINO: ${dtype}`);
    }
  }

  private uint8ArrayToString(arr: Uint8Array): string {
    // Fast arbitrary byte array to string for deduplication map keys
    let str = '';
    for (let i = 0; i < arr.length; i++) {
      str += String.fromCharCode(arr[i] || 0);
    }
    return str;
  }

  private emitShape(shape: (number | string)[], tagName: string = 'port'): XmlNode {
    const portNode = new XmlNode(tagName);
    for (let dim of shape) {
      let dimVal = dim.toString();
      if (dimVal === '-1' || (typeof dim === 'string' && isNaN(Number(dim)))) {
        dimVal = this.clampDynamic ? '1' : '-1';
      }
      const dimNode = new XmlNode('dim').addChild(dimVal);
      portNode.addChild(dimNode);
    }
    return portNode;
  }

  private addEdge(fromLayer: number, fromPort: number, toLayer: number, toPort: number) {
    const edge = new XmlNode('edge');
    edge.setAttribute('from-layer', fromLayer.toString());
    edge.setAttribute('from-port', fromPort.toString());
    edge.setAttribute('to-layer', toLayer.toString());
    edge.setAttribute('to-port', toPort.toString());

    for (const existing of this.edges) {
      if (
        existing.attributes['from-layer'] === fromLayer.toString() &&
        existing.attributes['from-port'] === fromPort.toString() &&
        existing.attributes['to-layer'] === toLayer.toString() &&
        existing.attributes['to-port'] === toPort.toString()
      ) {
        return;
      }
    }
    this.edges.push(edge);
  }

  export(): { xml: string; bin: Uint8Array } {
    const net = new XmlNode('net');
    net.setAttribute('name', this.graph.name || 'onnx9000_model');
    net.setAttribute('version', this.version);

    const layers = new XmlNode('layers');

    // Track consumed parameters to prevent emitting unused Parameters
    const consumedInputs = new Set<string>();
    for (const node of this.graph.nodes) {
      for (const inp of node.inputs) {
        consumedInputs.add(inp);
      }
    }

    // Map Parameters
    for (const valInfo of this.graph.inputs) {
      if (!consumedInputs.has(valInfo.name)) continue;

      const layerId = this.nextId();
      this.layerIds.set(valInfo.name, layerId);

      const layer = new XmlNode('layer');
      layer.setAttribute('id', layerId.toString());
      layer.setAttribute('name', valInfo.name);
      layer.setAttribute('type', 'Parameter');
      layer.setAttribute('version', 'opset1');

      const data = new XmlNode('data');
      const precisionStr = this.mapDtype(valInfo.dtype);
      data.setAttribute('element_type', precisionStr);
      const actualShape = valInfo.shape.length > 0 ? valInfo.shape : [1];
      data.setAttribute('shape', actualShape.join(','));
      layer.addChild(data);

      const outputPort = this.nextPort(layerId);
      const outNode = new XmlNode('output');
      const port = this.emitShape(actualShape, 'port');
      port.setAttribute('id', outputPort.toString());
      port.setAttribute('precision', precisionStr);
      outNode.addChild(port);
      layer.addChild(outNode);

      this.portIds.set(valInfo.name, { layerId, portId: outputPort });
      layers.addChild(layer);
    }

    // Map Constants
    for (const initName of this.graph.initializers) {
      const tensor = this.graph.tensors[initName];
      if (!tensor) continue;
      const layerId = this.nextId();
      this.layerIds.set(initName, layerId);

      const layer = new XmlNode('layer');
      layer.setAttribute('id', layerId.toString());
      layer.setAttribute('name', initName);
      layer.setAttribute('type', 'Const');
      layer.setAttribute('version', 'opset1');

      const data = new XmlNode('data');
      if (tensor.dtype) {
        data.setAttribute('element_type', this.mapDtype(tensor.dtype));
      }
      const actualShape = tensor.shape.length > 0 ? tensor.shape : [1];
      data.setAttribute('shape', actualShape.join(','));

      if (tensor.data && tensor.data.byteLength > 0) {
        let uint8View = new Uint8Array(
          tensor.data.buffer,
          tensor.data.byteOffset,
          tensor.data.byteLength,
        );

        if (this.compressToFp16 && tensor.dtype === 'float32') {
          // Global FP16 cast via float32 -> fp16 view map
          const f32 = new Float32Array(
            tensor.data.buffer,
            tensor.data.byteOffset,
            tensor.data.byteLength / 4,
          );
          const f16 = new Uint16Array(f32.length);
          for (let i = 0; i < f32.length; i++) {
            // Simple FP32 to FP16 converter
            const f = f32[i] || 0;
            const buffer = new ArrayBuffer(4);
            new Float32Array(buffer)[0] = f;
            const uint32 = new Uint32Array(buffer)[0] || 0;
            const sign = (uint32 >> 16) & 0x8000;
            const exponent = ((uint32 >> 23) & 0xff) - 127 + 15;
            let fraction = (uint32 >> 13) & 0x3ff;
            let e = exponent;
            if (exponent <= 0) {
              e = 0;
              fraction = 0;
            } else if (exponent >= 31) {
              e = 31;
              fraction = 0;
            }
            f16[i] = sign | (e << 10) | fraction;
          }
          uint8View = new Uint8Array(f16.buffer);
        }

        const hashKey = this.uint8ArrayToString(uint8View);
        const cacheHit = this.binCache.get(hashKey);

        if (cacheHit) {
          data.setAttribute('offset', cacheHit.offset.toString());
          data.setAttribute('size', cacheHit.size.toString());
        } else {
          const totalLength = this.binBuffer.reduce((acc, val) => acc + val.length, 0);
          this.binBuffer.push(uint8View);
          data.setAttribute('offset', totalLength.toString());
          data.setAttribute('size', uint8View.length.toString());
          this.binCache.set(hashKey, { offset: totalLength, size: uint8View.length });
        }
      } else {
        data.setAttribute('offset', '0');
        data.setAttribute('size', '0');
      }
      layer.addChild(data);

      const outputPort = this.nextPort(layerId);
      const outNode = new XmlNode('output');
      const port = this.emitShape(actualShape, 'port');
      port.setAttribute('id', outputPort.toString());
      if (tensor.dtype) {
        port.setAttribute('precision', this.mapDtype(tensor.dtype));
      }
      outNode.addChild(port);
      layer.addChild(outNode);

      this.portIds.set(initName, { layerId, portId: outputPort });
      layers.addChild(layer);
    }

    // Map Nodes
    for (const node of this.graph.nodes) {
      const layerId = this.nextId();
      const layerName = node.name || `${node.opType}_${layerId}`;
      this.layerIds.set(layerName, layerId);

      const layer = new XmlNode('layer');
      layer.setAttribute('id', layerId.toString());
      layer.setAttribute('name', layerName);
      layer.setAttribute('version', 'opset1');

      const typeMapping: Record<string, string> = {
        Sub: 'Subtract',
        Mul: 'Multiply',
        Div: 'Divide',
        Pow: 'Power',
        Max: 'Maximum',
        Min: 'Minimum',
        Ceil: 'Ceiling',
        Conv: 'Convolution',
        Relu: 'ReLU',
        LeakyRelu: 'PRelu',
        Sigmoid: 'Sigmoid',
        Tanh: 'Tanh',
        Elu: 'Elu',
        Selu: 'Selu',
        Softplus: 'SoftPlus',
        Gelu: 'Gelu',
        Softmax: 'SoftMax',
        LogSoftmax: 'LogSoftmax',
        PRelu: 'PRelu',
        Clip: 'Clamp',
        HardSigmoid: 'HardSigmoid',
        AveragePool: 'AvgPool',
        MaxPool: 'MaxPool',
        Flatten: 'Reshape',
        Reshape: 'Reshape',
        Transpose: 'Transpose',
        Squeeze: 'Squeeze',
        Unsqueeze: 'Unsqueeze',
        Concat: 'Concat',
        Split: 'Split',
        Gather: 'Gather',
        GatherND: 'GatherND',
        ScatterND: 'ScatterNDUpdate',
        ScatterElements: 'ScatterElementsUpdate',
        Shape: 'ShapeOf',
        Tile: 'Tile',
        Expand: 'Broadcast',
        ConstantOfShape: 'Broadcast',
        Cast: 'Convert',
        Pad: 'Pad',
        ReduceMean: 'ReduceMean',
        ReduceMax: 'ReduceMax',
        ReduceMin: 'ReduceMin',
        ReduceSum: 'ReduceSum',
        ReduceProd: 'ReduceProd',
        ArgMax: 'ArgMax',
        ArgMin: 'ArgMin',
        TopK: 'TopK',
        NonZero: 'NonZero',
        Equal: 'Equal',
        Not: 'LogicalNot',
        And: 'LogicalAnd',
        Or: 'LogicalOr',
        Xor: 'LogicalXor',
        Greater: 'Greater',
        Less: 'Less',
        GreaterOrEqual: 'GreaterEqual',
        LessOrEqual: 'LessEqual',
        Where: 'Select',
        Resize: 'Interpolate',
        SpaceToDepth: 'SpaceToDepth',
        DepthToSpace: 'DepthToSpace',
        NonMaxSuppression: 'NonMaxSuppression',
        RoiAlign: 'ROIAlign',
        CumSum: 'CumSum',
        QuantizeLinear: 'FakeQuantize',
        DequantizeLinear: 'FakeQuantize',
        If: 'If',
        Loop: 'TensorIterator',
        Scan: 'TensorIterator',
        Attention: 'ScaledDotProductAttention',
        Gemm: 'MatMul',
        Einsum: 'Einsum',
        Round: 'Round',
        BatchNormalization: 'BatchNormInference',
        InstanceNormalization: 'MVN',
        LayerNormalization: 'MVN',
        LpNormalization: 'NormalizeL2',
      };
      let ovType = typeMapping[node.opType] || node.opType;
      let hasDecoupledBias = false;
      let biasInpName = '';
      let inputsToMap = node.inputs;
      if (
        (node.opType === 'Conv' || node.opType === 'ConvTranspose' || node.opType === 'Gemm') &&
        node.inputs.length === 3
      ) {
        hasDecoupledBias = true;
        biasInpName = node.inputs[2] as string;
        inputsToMap = node.inputs.slice(0, 2);
      } else {
        inputsToMap = node.inputs.slice();
      }
      layer.setAttribute('type', ovType);

      const data = new XmlNode('data');

      const binaryOps = [
        'Add',
        'Subtract',
        'Multiply',
        'Divide',
        'Power',
        'Maximum',
        'Minimum',
        'Mod',
        'Equal',
        'Less',
        'Greater',
        'LessEqual',
        'GreaterEqual',
        'LogicalAnd',
        'LogicalOr',
        'LogicalXor',
      ];
      if (binaryOps.includes(ovType)) {
        data.setAttribute('auto_broadcast', 'numpy');
      }
      if (node.attributes) {
        if (node.opType === 'MatMul' || node.opType === 'Gemm') {
          const transA = node.attributes['transA']?.value ? 'true' : 'false';
          const transB = node.attributes['transB']?.value ? 'true' : 'false';
          data.setAttribute('transpose_a', transA);
          data.setAttribute('transpose_b', transB);
        } else if (node.opType === 'Conv') {
          if (node.attributes['strides']) {
            data.setAttribute('strides', (node.attributes['strides'].value as any[]).join(','));
          }
          if (node.attributes['dilations']) {
            data.setAttribute('dilations', (node.attributes['dilations'].value as any[]).join(','));
          }
          if (node.attributes['pads']) {
            const pads = node.attributes['pads'].value as any[];
            if (pads.length === 4) {
              data.setAttribute('pads_begin', `${pads[0]},${pads[1]}`);
              data.setAttribute('pads_end', `${pads[2]},${pads[3]}`);
            } else {
              const half = Math.floor(pads.length / 2);
              data.setAttribute('pads_begin', pads.slice(0, half).join(','));
              data.setAttribute('pads_end', pads.slice(half).join(','));
            }
          }
          if (node.attributes['auto_pad']) {
            const autoPadMap: Record<string, string> = {
              VALID: 'valid',
              SAME_UPPER: 'same_upper',
              SAME_LOWER: 'same_lower',
            };
            data.setAttribute(
              'auto_pad',
              autoPadMap[node.attributes['auto_pad'].value as string] || 'explicit',
            );
          }
        } else if (node.opType === 'MaxPool' || node.opType === 'AveragePool') {
          if (node.attributes['kernel_shape']) {
            data.setAttribute('kernel', (node.attributes['kernel_shape'].value as any[]).join(','));
          }
          if (node.attributes['strides']) {
            data.setAttribute('strides', (node.attributes['strides'].value as any[]).join(','));
          }
          if (node.attributes['pads']) {
            const pads = node.attributes['pads'].value as any[];
            if (pads.length === 4) {
              data.setAttribute('pads_begin', `${pads[0]},${pads[1]}`);
              data.setAttribute('pads_end', `${pads[2]},${pads[3]}`);
            } else {
              const half = Math.floor(pads.length / 2);
              data.setAttribute('pads_begin', pads.slice(0, half).join(','));
              data.setAttribute('pads_end', pads.slice(half).join(','));
            }
          }
          if (node.attributes['auto_pad']) {
            const autoPadMap: Record<string, string> = {
              VALID: 'valid',
              SAME_UPPER: 'same_upper',
              SAME_LOWER: 'same_lower',
            };
            data.setAttribute(
              'auto_pad',
              autoPadMap[node.attributes['auto_pad'].value as string] || 'explicit',
            );
          }
          if (node.opType === 'AveragePool' && node.attributes['count_include_pad']) {
            data.setAttribute(
              'exclude-pad',
              node.attributes['count_include_pad'].value ? 'false' : 'true',
            );
          }
        } else if (node.opType === 'Gelu') {
          const approx = node.attributes['approximate'];
          if (approx && approx.value === 'tanh') {
            data.setAttribute('approximation_mode', 'tanh');
          } else {
            data.setAttribute('approximation_mode', 'erf');
          }
        } else if (node.opType === 'Softmax') {
          const attr = node.attributes['axis'];
          if (attr && attr.value !== undefined && attr.value !== null) {
            data.setAttribute('axis', attr.value.toString());
          }
        } else if (node.opType === 'Concat' || node.opType === 'Split') {
          const attr = node.attributes['axis'];
          if (attr && attr.value !== undefined && attr.value !== null) {
            data.setAttribute('axis', attr.value.toString());
          }
        } else if (node.opType === 'Pad') {
          if (node.attributes && node.attributes['pads']) {
            const padsData = node.attributes['pads'].value as number[];
            const mid = Math.floor(padsData.length / 2);
            const padsBegin = padsData.slice(0, mid);
            const padsEnd = padsData.slice(mid);

            const bNode = this.emitDynamicConst(
              (node.name || `pads_begin_${layerId}`) + '_pads_begin',
              padsBegin,
              [padsBegin.length],
              'int64',
            );
            const eNode = this.emitDynamicConst(
              (node.name || `pads_end_${layerId}`) + '_pads_end',
              padsEnd,
              [padsEnd.length],
              'int64',
            );
            layers.addChild(bNode.layerNode);
            layers.addChild(eNode.layerNode);
            inputsToMap.push((node.name || `pads_begin_${layerId}`) + '_pads_begin');
            inputsToMap.push((node.name || `pads_end_${layerId}`) + '_pads_end');

            let val = 0.0;
            if (node.attributes['value']) {
              val = node.attributes['value'].value as number;
            }
            const vNode = this.emitDynamicConst(
              (node.name || `pad_value_${layerId}`) + '_pad_value',
              [val],
              [1],
              'float32',
            );
            layers.addChild(vNode.layerNode);
            inputsToMap.push((node.name || `pad_value_${layerId}`) + '_pad_value');
          } else if (node.inputs.length === 2) {
            const vNode = this.emitDynamicConst(
              (node.name || `pad_value_${layerId}`) + '_pad_value',
              [0.0],
              [1],
              'float32',
            );
            layers.addChild(vNode.layerNode);
            inputsToMap.push((node.name || `pad_value_${layerId}`) + '_pad_value');
          }
        } else if (node.opType === 'Gather') {
          const attr = node.attributes['batch_dims'];
          if (attr && attr.value !== undefined && attr.value !== null) {
            data.setAttribute('batch_dims', attr.value.toString());
          }
          if (node.attributes['axis'] && inputsToMap.length === 2) {
            const axisVal = node.attributes['axis'].value as number;
            const axisNode = this.emitDynamicConst(
              (node.name || `gather_axis_${layerId}`) + '_gather_axis',
              [axisVal],
              [1],
              'int64',
            );
            layers.addChild(axisNode.layerNode);
            inputsToMap.push((node.name || `gather_axis_${layerId}`) + '_gather_axis');
          }
        } else if (node.opType === 'Slice') {
          ovType = 'StridedSlice';
          layer.setAttribute('type', 'StridedSlice');
          data.setAttribute('begin_mask', '');
          data.setAttribute('end_mask', '');
          data.setAttribute('new_axis_mask', '');
          data.setAttribute('shrink_axis_mask', '');
          data.setAttribute('ellipsis_mask', '');
        } else if (
          ['ReduceMean', 'ReduceMax', 'ReduceMin', 'ReduceSum', 'ReduceProd'].includes(node.opType)
        ) {
          const attr = node.attributes['keepdims'];
          if (attr && attr.value !== undefined && attr.value !== null) {
            data.setAttribute('keep_dims', attr.value ? 'true' : 'false');
          }
        } else if (['ArgMax', 'ArgMin'].includes(node.opType)) {
          const keep = node.attributes['keepdims'];
          if (keep && keep.value !== undefined && keep.value !== null) {
            data.setAttribute('keep_dims', keep.value ? 'true' : 'false');
          }
          const axis = node.attributes['axis'];
          if (axis && axis.value !== undefined && axis.value !== null) {
            data.setAttribute('axis', axis.value.toString());
          }
        } else if (node.opType === 'Resize') {
          const mode = node.attributes['mode'];
          if (mode && mode.value !== undefined && mode.value !== null) {
            data.setAttribute('mode', mode.value.toString());
          }
          const coordMode = node.attributes['coordinate_transformation_mode'];
          if (coordMode && coordMode.value !== undefined && coordMode.value !== null) {
            data.setAttribute('coordinate_transformation_mode', coordMode.value.toString());
          }
          data.setAttribute('shape_calculation_mode', 'sizes');
          const nearestMode = node.attributes['nearest_mode'];
          if (nearestMode && nearestMode.value !== undefined && nearestMode.value !== null) {
            data.setAttribute('nearest_mode', nearestMode.value.toString());
          }
        } else if (['SpaceToDepth', 'DepthToSpace'].includes(node.opType)) {
          const blocksize = node.attributes['blocksize'];
          if (blocksize && blocksize.value !== undefined && blocksize.value !== null) {
            data.setAttribute('block_size', blocksize.value.toString());
          }
          const mode = node.attributes['mode'];
          if (mode && mode.value !== undefined && mode.value !== null) {
            data.setAttribute('mode', mode.value.toString());
          }
        } else if (node.opType === 'NonMaxSuppression') {
          if (node.attributes && node.attributes['center_point_box']) {
            const val = node.attributes['center_point_box'].value;
            data.setAttribute('box_encoding', val ? 'center' : 'corner');
          } else {
            data.setAttribute('box_encoding', 'corner');
          }
          data.setAttribute('sort_result_descending', 'false');
        } else if (node.opType === 'RoiAlign') {
          if (node.attributes && node.attributes['mode']) {
            data.setAttribute('mode', (node.attributes['mode']?.value as any)?.toString());
          } else {
            data.setAttribute('mode', 'avg');
          }
        } else if (node.opType === 'QuantizeLinear' || node.opType === 'DequantizeLinear') {
          data.setAttribute('levels', '256');
        } else if (node.opType === 'Einsum') {
          const eq = node.attributes['equation'];
          if (eq && eq.value !== undefined && eq.value !== null) {
            data.setAttribute('equation', eq.value.toString());
          }
        } else if (
          node.opType === 'LayerNormalization' ||
          node.opType === 'InstanceNormalization'
        ) {
          const eps = node.attributes['epsilon'];
          if (eps && eps.value !== undefined && eps.value !== null) {
            data.setAttribute('eps', eps.value.toString());
          }
          if (node.opType === 'LayerNormalization') {
            const axis = node.attributes['axis'];
            if (axis && axis.value !== undefined && axis.value !== null) {
              data.setAttribute('axes', axis.value.toString());
            }
          }
          data.setAttribute('normalize_variance', 'true');
          data.setAttribute('eps_mode', 'add');
        } else if (node.opType === 'LpNormalization') {
          const axis = node.attributes['axis'];
          if (axis && axis.value !== undefined && axis.value !== null) {
            data.setAttribute('axes', axis.value.toString());
          }
          const p = node.attributes['p'];
          if (p && p.value !== undefined && p.value !== null) {
            data.setAttribute('p', p.value.toString());
          }
        } else if (node.opType === 'BatchNormalization') {
          const eps = node.attributes['epsilon'];
          if (eps && eps.value !== undefined && eps.value !== null) {
            data.setAttribute('eps', eps.value.toString());
          }
        } else if (node.opType === 'Dropout') {
          // Handled implicitly by graph surgeon usually
        } else if (node.opType === 'If') {
          if (node.attributes['then_branch']) {
            const subGraph = node.attributes['then_branch'].value as Graph;
            const subExporter = new OpenVinoExporter(subGraph, {
              version: this.version,
              compressToFp16: this.compressToFp16,
            });
            subExporter.export();
            new XmlNode('body');
          }
          if (node.attributes['else_branch']) {
            const elseGraph = node.attributes['else_branch'].value as Graph;
            const elseExporter = new OpenVinoExporter(elseGraph, {
              version: this.version,
              compressToFp16: this.compressToFp16,
            });
            elseExporter.export();
            new XmlNode('body');
          }
        } else if (node.opType === 'Loop' || node.opType === 'Scan') {
          if (node.attributes['body']) {
            const subGraph = node.attributes['body'].value as Graph;
            const subExporter = new OpenVinoExporter(subGraph, {
              version: this.version,
              compressToFp16: this.compressToFp16,
            });
            subExporter.export();
            new XmlNode('body');
          }
        }
      }

      // Always add data block for FakeQuantize
      if (Object.keys(data.attributes).length > 0 || ovType === 'FakeQuantize') {
        if (ovType === 'FakeQuantize' && !data.attributes['levels']) {
          data.setAttribute('levels', '256');
        }
        layer.addChild(data);
      }

      if (node.opType === 'Cast') {
        ovType = 'Convert';
        layer.setAttribute('type', 'Convert');
        if (node.attributes && node.attributes['to']) {
          const toDtype = node.attributes['to'].value as string;
          data.setAttribute('destination_type', this.mapDtype(toDtype as DType));
        }
      } else if (node.opType === 'GridSample') {
        ovType = 'GridSample';
        layer.setAttribute('type', 'GridSample');
        if (node.attributes && node.attributes['mode']) {
          data.setAttribute('mode', (node.attributes['mode']?.value as any)?.toString());
        }
        if (node.attributes && node.attributes['padding_mode']) {
          data.setAttribute(
            'padding_mode',
            (node.attributes['padding_mode']?.value as any)?.toString(),
          );
        }
        if (node.attributes && node.attributes['align_corners']) {
          data.setAttribute(
            'align_corners',
            node.attributes['align_corners'].value ? 'true' : 'false',
          );
        }
      } else if (node.opType === 'Size') {
        const shapeLayerId = this.nextId();
        const shapeName = (node.name || `shapeof_${layerId}`) + '_shapeof';
        const shapeLayer = new XmlNode('layer')
          .setAttribute('id', shapeLayerId.toString())
          .setAttribute('name', shapeName)
          .setAttribute('type', 'ShapeOf')
          .setAttribute('version', 'opset1');

        const shapeInNode = new XmlNode('input');
        const shapeInPort = this.nextPort(shapeLayerId);
        shapeInNode.addChild(new XmlNode('port').setAttribute('id', shapeInPort.toString()));
        shapeLayer.addChild(shapeInNode);

        if (node.inputs.length > 0) {
          const fromIds = this.portIds.get(node.inputs[0] as string);
          if (fromIds) {
            this.addEdge(fromIds.layerId, fromIds.portId, shapeLayerId, shapeInPort);
          }
        }

        const shapeOutNode = new XmlNode('output');
        const shapeOutPort = this.nextPort(shapeLayerId);
        shapeOutNode.addChild(
          new XmlNode('port')
            .setAttribute('id', shapeOutPort.toString())
            .setAttribute('precision', 'i64'),
        );
        shapeLayer.addChild(shapeOutNode);

        layers.addChild(shapeLayer);

        const axesConst = this.emitDynamicConst(
          (node.name || `axes_${layerId}`) + '_axes',
          [0],
          [1],
          'int64',
        );
        layers.addChild(axesConst.layerNode);

        ovType = 'ReduceProd';
        layer.setAttribute('type', 'ReduceProd');
        data.setAttribute('keep_dims', 'false');

        inputsToMap = [shapeName, (node.name || `axes_${layerId}`) + '_axes'];
        this.portIds.set(shapeName, { layerId: shapeLayerId, portId: shapeOutPort });
      } else if (node.opType === 'Flatten') {
        const shapeData = [0, -1];
        const constNode = this.emitDynamicConst(
          (node.name || `flatten_shape_${layerId}`) + '_flatten_shape',
          shapeData,
          [2],
          'int64',
        );
        layers.addChild(constNode.layerNode);
        inputsToMap.push((node.name || `flatten_shape_${layerId}`) + '_flatten_shape');
      } else if (node.opType === 'Transpose' && node.attributes && node.attributes['perm']) {
        const permData = node.attributes['perm'].value as number[];
        const constNode = this.emitDynamicConst(
          (node.name || `transpose_perm_${layerId}`) + '_transpose_perm',
          permData,
          [permData.length],
          'int64',
        );
        layers.addChild(constNode.layerNode);
        inputsToMap.push((node.name || `transpose_perm_${layerId}`) + '_transpose_perm');
      } else if (
        ['ReduceMean', 'ReduceMax', 'ReduceMin', 'ReduceSum', 'ReduceProd'].includes(node.opType)
      ) {
        if (node.attributes && node.attributes['axes']) {
          const axesData = node.attributes['axes'].value as number[];
          const constNode = this.emitDynamicConst(
            (node.name || `reduce_axes_${layerId}`) + '_reduce_axes',
            axesData,
            [axesData.length],
            'int64',
          );
          layers.addChild(constNode.layerNode);
          inputsToMap.push((node.name || `reduce_axes_${layerId}`) + '_reduce_axes');
        }
      } else if (node.opType === 'GatherElements') {
        ovType = 'GatherElements';
        layer.setAttribute('type', 'GatherElements');
        if (node.attributes && node.attributes['axis']) {
          const axisVal = node.attributes['axis'].value as number;
          const axisNode = this.emitDynamicConst(
            (node.name || `gather_el_axis_${layerId}`) + '_gather_el_axis',
            [axisVal],
            [1],
            'int64',
          );
          layers.addChild(axisNode.layerNode);
          inputsToMap.push((node.name || `gather_el_axis_${layerId}`) + '_gather_el_axis');
        }
      } else if (node.opType === 'ConstantOfShape') {
        let val = [0.0];
        let dtype: DType = 'float32';
        if (node.attributes && node.attributes['value']) {
          const valTensor = node.attributes['value'].value as any;
          if (valTensor.dtype === 'float32') {
            val = [new Float32Array(valTensor.data.buffer, valTensor.data.byteOffset, 1)[0] || 0.0];
            dtype = 'float32';
          } else if (valTensor.dtype === 'int64') {
            const bi =
              new BigInt64Array(valTensor.data.buffer, valTensor.data.byteOffset, 1)[0] || 0n;
            val = [Number(bi)];
            dtype = 'int64';
          } else if (valTensor.dtype === 'int32') {
            val = [new Int32Array(valTensor.data.buffer, valTensor.data.byteOffset, 1)[0] || 0];
            dtype = 'int32';
          }
        }

        const constNode = this.emitDynamicConst(
          (node.name || `scalar_val_${layerId}`) + '_scalar_val',
          val,
          [1],
          dtype,
        );
        layers.addChild(constNode.layerNode);
        inputsToMap = [
          (node.name || `scalar_val_${layerId}`) + '_scalar_val',
          node.inputs[0] as string,
        ];
        ovType = 'Broadcast';
      }

      const inNode = new XmlNode('input');
      for (const inp of inputsToMap) {
        const inputPort = this.nextPort(layerId);
        const port = new XmlNode('port');
        port.setAttribute('id', inputPort.toString());
        inNode.addChild(port);

        const fromIds = this.portIds.get(inp as string);
        if (fromIds) {
          this.addEdge(fromIds.layerId, fromIds.portId, layerId, inputPort);
        } else if (inp !== '') {
          throw new Error(`Missing input pointer: '${inp}' for node '${node.name || layerId}'`);
        }
      }
      if (inputsToMap.length > 0) {
        layer.addChild(inNode);
      }

      const outNode = new XmlNode('output');
      for (const out of node.outputs) {
        const outputPort = this.nextPort(layerId);
        const port = new XmlNode('port');
        port.setAttribute('id', outputPort.toString());
        outNode.addChild(port);

        if (hasDecoupledBias) {
          this.portIds.set(out + '_internal_nobias', { layerId, portId: outputPort });
        } else {
          this.portIds.set(out, { layerId, portId: outputPort });
        }
      }
      if (node.outputs.length > 0) {
        layer.addChild(outNode);
      }

      layers.addChild(layer);

      if (hasDecoupledBias) {
        for (const out of node.outputs) {
          const addLayerId = this.nextId();
          const addLayer = new XmlNode('layer');
          addLayer.setAttribute('id', addLayerId.toString());
          addLayer.setAttribute('name', out + '_bias_add');
          addLayer.setAttribute('type', 'Add');
          addLayer.setAttribute('version', 'opset1');

          const addData = new XmlNode('data').setAttribute('auto_broadcast', 'numpy');
          addLayer.addChild(addData);

          const addInNode = new XmlNode('input');

          const p1 = this.nextPort(addLayerId);
          addInNode.addChild(new XmlNode('port').setAttribute('id', p1.toString()));
          const nobiasIds = this.portIds.get(out + '_internal_nobias');
          if (nobiasIds) {
            this.addEdge(nobiasIds.layerId, nobiasIds.portId, addLayerId, p1);
          }

          const p2 = this.nextPort(addLayerId);
          addInNode.addChild(new XmlNode('port').setAttribute('id', p2.toString()));
          const biasIds = this.portIds.get(biasInpName);
          if (biasIds) {
            this.addEdge(biasIds.layerId, biasIds.portId, addLayerId, p2);
          }

          addLayer.addChild(addInNode);

          const addOutNode = new XmlNode('output');
          const p3 = this.nextPort(addLayerId);
          addOutNode.addChild(new XmlNode('port').setAttribute('id', p3.toString()));
          addLayer.addChild(addOutNode);

          this.portIds.set(out, { layerId: addLayerId, portId: p3 });

          layers.addChild(addLayer);
        }
      }
    }

    // Map Results
    for (const valInfo of this.graph.outputs) {
      const fromIds = this.portIds.get(valInfo.name);
      if (!fromIds) continue;

      const layerId = this.nextId();
      this.layerIds.set(valInfo.name + '_result', layerId);

      const layer = new XmlNode('layer');
      layer.setAttribute('id', layerId.toString());
      layer.setAttribute('name', valInfo.name + '_result');
      layer.setAttribute('type', 'Result');
      layer.setAttribute('version', 'opset1');

      const inputPort = this.nextPort(layerId);
      const inNode = new XmlNode('input');
      const port = this.emitShape(valInfo.shape, 'port');
      port.setAttribute('id', inputPort.toString());
      port.setAttribute('precision', this.mapDtype(valInfo.dtype));
      inNode.addChild(port);
      layer.addChild(inNode);

      layers.addChild(layer);

      this.addEdge(fromIds.layerId, fromIds.portId, layerId, inputPort);
    }
    net.addChild(layers);

    const edgesNode = new XmlNode('edges');
    for (const e of this.edges) {
      edgesNode.addChild(e);
    }
    net.addChild(edgesNode);

    const rtInfo = new XmlNode('rt_info');
    const metaData = new XmlNode('meta_data');
    const moSettings = new XmlNode('MO_version').setAttribute('value', 'onnx9000');
    const conversionParams = new XmlNode('cli_parameters');
    conversionParams.addChild(
      new XmlNode('compress_to_fp16').setAttribute('value', this.compressToFp16.toString()),
    );
    metaData.addChild(moSettings);
    metaData.addChild(conversionParams);
    rtInfo.addChild(metaData);
    net.addChild(rtInfo);

    const builder = new XmlBuilder();
    builder.setRoot(net);
    const xmlStr = builder.toString(true);

    const totalLength = this.binBuffer.reduce((acc, val) => acc + val.length, 0);
    const binArray = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of this.binBuffer) {
      binArray.set(chunk, offset);
      offset += chunk.length;
    }

    return { xml: xmlStr, bin: binArray };
  }
}
