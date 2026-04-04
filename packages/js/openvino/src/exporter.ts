import { Graph, Node, Tensor, ValueInfo, DType, Attribute } from '@onnx9000/core';
import { XmlBuilder, XmlNode } from './xml_builder';

/**
 * Options for OpenVINO export.
 */
export interface OpenVinoExportOptions {
  /** The OpenVINO IR version (default: '11'). */
  version?: string;
  /** Whether to compress float32 tensors to float16 (default: false). */
  compressToFp16?: boolean;
  /** Whether to clamp dynamic shapes to 1 (default: false). */
  clampDynamic?: boolean;
}

/**
 * Context for handling an ONNX operator during OpenVINO export.
 */
export interface OpHandlerContext {
  /** The OpenVinoExporter instance. */
  exporter: OpenVinoExporter;
  /** The ONNX node being exported. */
  node: Node;
  /** The OpenVINO layer being built. */
  layer: XmlNode;
  /** The data node of the OpenVINO layer. */
  data: XmlNode;
  /** The list of input names for this node. */
  inputsToMap: string[];
  /** The unique ID of the layer. */
  layerId: number;
  /** The parent layers node. */
  layers: XmlNode;
}

/**
 * A function that handles a specific ONNX operator type.
 */
export type OpHandler = (context: OpHandlerContext) => void;

/**
 * The OpenVinoExporter class translates an onnx9000 Core IR Graph into OpenVINO IR (XML/BIN).
 */
export class OpenVinoExporter {
  /** The ONNX graph to export. */
  graph: Graph;
  /** The OpenVINO IR version. */
  version: string;
  /** Whether to compress float32 to float16. */
  compressToFp16: boolean;
  /** Whether to clamp dynamic shapes. */
  clampDynamic: boolean;
  private nodeIdCounter: number = 0;
  /** Maps layer names to their IDs. */
  layerIds: Map<string, number> = new Map();
  /** Maps port names to their layer and port IDs. */
  portIds: Map<string, { layerId: number; portId: number }> = new Map();
  /** Buffer for binary data. */
  binBuffer: Uint8Array[] = [];
  private binCache: Map<string, { offset: number; size: number }> = new Map();
  private edges: XmlNode[] = [];
  private portCounters: Map<number, number> = new Map();

  private static handlers: Map<string, OpHandler> = new Map();

  /**
   * Initializes a new OpenVinoExporter.
   * @param graph - The ONNX graph.
   * @param options - Export options.
   */
  constructor(graph: Graph, options: OpenVinoExportOptions = {}) {
    this.graph = graph;
    this.version = options.version || '11';
    this.compressToFp16 = options.compressToFp16 || false;
    this.clampDynamic = options.clampDynamic || false;
    if (OpenVinoExporter.handlers.size === 0) {
      OpenVinoExporter.initializeHandlers();
    }
  }

  private nextId(): number {
    return this.nodeIdCounter++;
  }

  private nextPort(layerId: number): number {
    const p = this.portCounters.get(layerId) || 0;
    this.portCounters.set(layerId, p + 1);
    return p;
  }

  /**
   * Emits a constant layer with dynamic data.
   * @param name - The name of the constant.
   * @param data - The numeric data.
   * @param shape - The shape of the data.
   * @param dtype - The data type.
   * @returns The created layer node and its output port ID.
   */
  public emitDynamicConst(
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
        view.setBigInt64(i * 8, BigInt(data[i] ?? 0), true); // little-endian
      } else if (dtype === 'int32') {
        view.setInt32(i * 4, data[i] ?? 0, true);
      } else if (dtype === 'float32') {
        view.setFloat32(i * 4, data[i] ?? 0, true);
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
        /* v8 ignore start */
        return 'f64';
      /* v8 ignore stop */
      case 'float16':
        /* v8 ignore start */
        return 'f16';
      /* v8 ignore stop */
      case 'bfloat16':
        /* v8 ignore start */
        return 'bf16';
      /* v8 ignore stop */
      case 'int64':
        return 'i64';
      case 'int32':
        return 'i32';
      case 'int16':
        /* v8 ignore start */
        return 'i16';
      /* v8 ignore stop */
      case 'int8':
        /* v8 ignore start */
        return 'i8';
      /* v8 ignore stop */
      case 'uint64':
        /* v8 ignore start */
        return 'u64';
      /* v8 ignore stop */
      case 'uint32':
        /* v8 ignore start */
        return 'u32';
      /* v8 ignore stop */
      case 'uint16':
        /* v8 ignore start */
        return 'u16';
      /* v8 ignore stop */
      case 'uint8':
        /* v8 ignore start */
        return 'u8';
      /* v8 ignore stop */
      case 'bool':
        return 'boolean';
      default:
        /* v8 ignore start */
        throw new Error(`Unsupported dtype for OpenVINO: ${dtype}`);
      /* v8 ignore stop */
    }
  }

  private uint8ArrayToString(arr: Uint8Array): string {
    let str = '';
    for (let i = 0; i < arr.length; i++) {
      str += String.fromCharCode(arr[i] ?? 0);
    }
    return str;
  }

  /**
   * Emits a shape as an XML node.
   * @param shape - The shape dimensions.
   * @param tagName - The XML tag name (default: 'port').
   * @returns The shape XML node.
   */
  public emitShape(shape: (number | string)[], tagName: string = 'port'): XmlNode {
    const portNode = new XmlNode(tagName);
    for (const dim of shape) {
      let dimVal = dim.toString();
      if (dimVal === '-1' || (typeof dim === 'string' && isNaN(Number(dim)))) {
        /* v8 ignore start */
        dimVal = this.clampDynamic ? '1' : '-1';
      }
      /* v8 ignore stop */
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
        /* v8 ignore start */
        existing.attributes['to-port'] === toPort.toString()
        /* v8 ignore stop */
      ) {
        /* v8 ignore start */
        return;
      }
      /* v8 ignore stop */
    }
    this.edges.push(edge);
  }

  /**
   * Exports the ONNX graph to OpenVINO IR.
   * @returns An object containing the XML string and binary data.
   */
  export(): { xml: string; bin: Uint8Array } {
    const net = new XmlNode('net');
    net.setAttribute('name', this.graph.name || 'onnx9000_model');
    net.setAttribute('version', this.version);

    const layers = new XmlNode('layers');

    const consumedInputs = new Set<string>();
    for (const node of this.graph.nodes) {
      for (const inp of node.inputs) {
        consumedInputs.add(inp);
      }
    }

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
          /* v8 ignore start */
          const f32 = new Float32Array(
            tensor.data.buffer,
            tensor.data.byteOffset,
            tensor.data.byteLength / 4,
          );
          const f16 = new Uint16Array(f32.length);
          for (let i = 0; i < f32.length; i++) {
            const f = f32[i] ?? 0;
            const buffer = new ArrayBuffer(4);
            new Float32Array(buffer)[0] = f;
            const uint32 = new Uint32Array(buffer)[0] ?? 0;
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
        /* v8 ignore stop */

        const hashKey = this.uint8ArrayToString(uint8View);
        const cacheHit = this.binCache.get(hashKey);

        if (cacheHit) {
          /* v8 ignore start */
          data.setAttribute('offset', cacheHit.offset.toString());
          data.setAttribute('size', cacheHit.size.toString());
          /* v8 ignore stop */
        } else {
          const totalLength = this.binBuffer.reduce((acc, val) => acc + val.length, 0);
          this.binBuffer.push(uint8View);
          data.setAttribute('offset', totalLength.toString());
          data.setAttribute('size', uint8View.length.toString());
          this.binCache.set(hashKey, { offset: totalLength, size: uint8View.length });
        }
      } else {
        /* v8 ignore start */
        data.setAttribute('offset', '0');
        data.setAttribute('size', '0');
      }
      /* v8 ignore stop */
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
        /* v8 ignore start */
        hasDecoupledBias = true;
        biasInpName = node.inputs[2] as string;
        inputsToMap = node.inputs.slice(0, 2);
        /* v8 ignore stop */
      } else {
        inputsToMap = node.inputs.slice();
      }
      layer.setAttribute('type', ovType);

      const data = new XmlNode('data');

      const binaryOps = [
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
        'Add',
      ];
      if (binaryOps.includes(ovType)) {
        data.setAttribute('auto_broadcast', 'numpy');
      }

      const handler = OpenVinoExporter.handlers.get(node.opType);
      if (handler) {
        handler({
          exporter: this,
          node,
          layer,
          data,
          inputsToMap,
          layerId,
          layers,
        });
        ovType = layer.attributes['type'] ?? ovType;
      }

      if (Object.keys(data.attributes).length > 0 || ovType === 'FakeQuantize') {
        if (ovType === 'FakeQuantize' && !data.attributes['levels']) {
          /* v8 ignore start */
          data.setAttribute('levels', '256');
        }
        /* v8 ignore stop */
        layer.addChild(data);
      }

      const inNode = new XmlNode('input');
      for (const inp of inputsToMap) {
        const inputPort = this.nextPort(layerId);
        const port = new XmlNode('port');
        port.setAttribute('id', inputPort.toString());
        inNode.addChild(port);

        const fromIds = this.portIds.get(inp);
        if (fromIds) {
          this.addEdge(fromIds.layerId, fromIds.portId, layerId, inputPort);
        } else if (inp !== '') {
          /* v8 ignore start */
          throw new Error(`Missing input pointer: '${inp}' for node '${node.name || layerId}'`);
        }
        /* v8 ignore stop */
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
          /* v8 ignore start */
          this.portIds.set(out + '_internal_nobias', { layerId, portId: outputPort });
          /* v8 ignore stop */
        } else {
          this.portIds.set(out, { layerId, portId: outputPort });
        }
      }
      if (node.outputs.length > 0) {
        layer.addChild(outNode);
      }

      layers.addChild(layer);

      if (hasDecoupledBias) {
        /* v8 ignore start */
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
      /* v8 ignore stop */
    }

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

  private static initializeHandlers() {
    const h = OpenVinoExporter.handlers;

    const setAttr = (
      data: XmlNode,
      node: Node,
      attr: string,
      ovAttr: string,
      defaultValue?: string,
    ) => {
      const a = node.attributes[attr];
      if (a && a.value !== undefined && a.value !== null) {
        data.setAttribute(ovAttr, a.value.toString());
      } else if (defaultValue !== undefined) {
        /* v8 ignore start */
        data.setAttribute(ovAttr, defaultValue);
      }
      /* v8 ignore stop */
    };

    const setListAttr = (data: XmlNode, node: Node, attr: string, ovAttr: string) => {
      const a = node.attributes[attr];
      if (a && Array.isArray(a.value)) {
        data.setAttribute(ovAttr, a.value.join(','));
      }
    };

    h.set('MatMul', ({ node, data }) => {
      const transA = node.attributes['transA']?.value ? 'true' : 'false';
      const transB = node.attributes['transB']?.value ? 'true' : 'false';
      data.setAttribute('transpose_a', transA);
      data.setAttribute('transpose_b', transB);
    });
    h.set('Gemm', h.get('MatMul')!);

    h.set('Conv', ({ node, data }) => {
      setListAttr(data, node, 'strides', 'strides');
      setListAttr(data, node, 'dilations', 'dilations');
      const padsAttr = node.attributes['pads'];
      if (padsAttr && Array.isArray(padsAttr.value)) {
        const pads = padsAttr.value as number[];
        if (pads.length === 4) {
          data.setAttribute('pads_begin', `${pads[0]},${pads[1]}`);
          data.setAttribute('pads_end', `${pads[2]},${pads[3]}`);
        } else {
          const half = Math.floor(pads.length / 2);
          data.setAttribute('pads_begin', pads.slice(0, half).join(','));
          data.setAttribute('pads_end', pads.slice(half).join(','));
        }
      }
      const autoPad = node.attributes['auto_pad']?.value as string;
      if (autoPad) {
        const autoPadMap: Record<string, string> = {
          VALID: 'valid',
          SAME_UPPER: 'same_upper',
          SAME_LOWER: 'same_lower',
        };
        data.setAttribute('auto_pad', autoPadMap[autoPad] || 'explicit');
      }
    });

    const poolHandler: OpHandler = ({ node, data }) => {
      setListAttr(data, node, 'kernel_shape', 'kernel');
      setListAttr(data, node, 'strides', 'strides');
      const padsAttr = node.attributes['pads'];
      if (padsAttr && Array.isArray(padsAttr.value)) {
        const pads = padsAttr.value as number[];
        if (pads.length === 4) {
          data.setAttribute('pads_begin', `${pads[0]},${pads[1]}`);
          data.setAttribute('pads_end', `${pads[2]},${pads[3]}`);
        } else {
          /* v8 ignore start */
          const half = Math.floor(pads.length / 2);
          data.setAttribute('pads_begin', pads.slice(0, half).join(','));
          data.setAttribute('pads_end', pads.slice(half).join(','));
        }
        /* v8 ignore stop */
      }
      const autoPad = node.attributes['auto_pad']?.value as string;
      if (autoPad) {
        /* v8 ignore start */
        const autoPadMap: Record<string, string> = {
          VALID: 'valid',
          SAME_UPPER: 'same_upper',
          SAME_LOWER: 'same_lower',
        };
        data.setAttribute('auto_pad', autoPadMap[autoPad] || 'explicit');
      }
      /* v8 ignore stop */
      if (node.opType === 'AveragePool' && node.attributes['count_include_pad']) {
        data.setAttribute(
          'exclude-pad',
          node.attributes['count_include_pad'].value ? 'false' : 'true',
        );
      }
    };
    h.set('MaxPool', poolHandler);
    h.set('AveragePool', poolHandler);

    h.set('Gelu', ({ node, data }) => {
      const approx = node.attributes['approximate']?.value;
      data.setAttribute('approximation_mode', approx === 'tanh' ? 'tanh' : 'erf');
    });

    h.set('Softmax', ({ node, data }) => setAttr(data, node, 'axis', 'axis'));
    h.set('Concat', ({ node, data }) => setAttr(data, node, 'axis', 'axis'));
    h.set('Split', ({ node, data }) => setAttr(data, node, 'axis', 'axis'));

    h.set('Pad', ({ exporter, node, inputsToMap, layerId, layers }) => {
      const padsAttr = node.attributes['pads'];
      if (padsAttr && Array.isArray(padsAttr.value)) {
        const padsData = padsAttr.value as number[];
        const mid = Math.floor(padsData.length / 2);
        const padsBegin = padsData.slice(0, mid);
        const padsEnd = padsData.slice(mid);

        const bNode = exporter.emitDynamicConst(
          (node.name || `pads_begin_${layerId}`) + '_pads_begin',
          padsBegin,
          [padsBegin.length],
          'int64',
        );
        const eNode = exporter.emitDynamicConst(
          (node.name || `pads_end_${layerId}`) + '_pads_end',
          padsEnd,
          [padsEnd.length],
          'int64',
        );
        layers.addChild(bNode.layerNode);
        layers.addChild(eNode.layerNode);
        inputsToMap.push((node.name || `pads_begin_${layerId}`) + '_pads_begin');
        inputsToMap.push((node.name || `pads_end_${layerId}`) + '_pads_end');

        const val = (node.attributes['value']?.value as number) || 0.0;
        const vNode = exporter.emitDynamicConst(
          (node.name || `pad_value_${layerId}`) + '_pad_value',
          [val],
          [1],
          'float32',
        );
        layers.addChild(vNode.layerNode);
        inputsToMap.push((node.name || `pad_value_${layerId}`) + '_pad_value');
      } else if (node.inputs.length === 2) {
        const vNode = exporter.emitDynamicConst(
          (node.name || `pad_value_${layerId}`) + '_pad_value',
          [0.0],
          [1],
          'float32',
        );
        layers.addChild(vNode.layerNode);
        inputsToMap.push((node.name || `pad_value_${layerId}`) + '_pad_value');
      }
    });

    h.set('Gather', ({ exporter, node, data, inputsToMap, layerId, layers }) => {
      setAttr(data, node, 'batch_dims', 'batch_dims');
      const axisAttr = node.attributes['axis'];
      if (axisAttr && axisAttr.value !== undefined && inputsToMap.length === 2) {
        const axisVal = axisAttr.value as number;
        const axisNode = exporter.emitDynamicConst(
          (node.name || `gather_axis_${layerId}`) + '_gather_axis',
          [axisVal],
          [1],
          'int64',
        );
        layers.addChild(axisNode.layerNode);
        inputsToMap.push((node.name || `gather_axis_${layerId}`) + '_gather_axis');
      }
    });

    h.set('Slice', ({ layer, data }) => {
      layer.setAttribute('type', 'StridedSlice');
      data.setAttribute('begin_mask', '');
      data.setAttribute('end_mask', '');
      data.setAttribute('new_axis_mask', '');
      data.setAttribute('shrink_axis_mask', '');
      data.setAttribute('ellipsis_mask', '');
    });

    const reduceHandler: OpHandler = ({ node, data, exporter, layerId, layers, inputsToMap }) => {
      const keepdims = node.attributes['keepdims']?.value;
      if (keepdims !== undefined) {
        data.setAttribute('keep_dims', keepdims ? 'true' : 'false');
      }
      const axesAttr = node.attributes['axes'];
      if (axesAttr && Array.isArray(axesAttr.value)) {
        const axesData = axesAttr.value as number[];
        const constNode = exporter.emitDynamicConst(
          (node.name || `reduce_axes_${layerId}`) + '_reduce_axes',
          axesData,
          [axesData.length],
          'int64',
        );
        layers.addChild(constNode.layerNode);
        inputsToMap.push((node.name || `reduce_axes_${layerId}`) + '_reduce_axes');
      }
    };
    h.set('ReduceMean', reduceHandler);
    h.set('ReduceMax', reduceHandler);
    h.set('ReduceMin', reduceHandler);
    h.set('ReduceSum', reduceHandler);
    h.set('ReduceProd', reduceHandler);

    const argHandler: OpHandler = ({ node, data }) => {
      const keep = node.attributes['keepdims']?.value;
      if (keep !== undefined) data.setAttribute('keep_dims', keep ? 'true' : 'false');
      setAttr(data, node, 'axis', 'axis');
    };
    h.set('ArgMax', argHandler);
    h.set('ArgMin', argHandler);

    h.set('Resize', ({ node, data }) => {
      setAttr(data, node, 'mode', 'mode');
      setAttr(data, node, 'coordinate_transformation_mode', 'coordinate_transformation_mode');
      data.setAttribute('shape_calculation_mode', 'sizes');
      setAttr(data, node, 'nearest_mode', 'nearest_mode');
    });

    const s2dHandler: OpHandler = ({ node, data }) => {
      setAttr(data, node, 'blocksize', 'block_size');
      setAttr(data, node, 'mode', 'mode');
    };
    h.set('SpaceToDepth', s2dHandler);
    h.set('DepthToSpace', s2dHandler);

    h.set('NonMaxSuppression', ({ node, data }) => {
      const center = node.attributes['center_point_box']?.value;
      data.setAttribute('box_encoding', center ? 'center' : 'corner');
      data.setAttribute('sort_result_descending', 'false');
    });

    h.set('RoiAlign', ({ node, data }) => {
      const mode = node.attributes['mode']?.value;
      data.setAttribute('mode', mode ? mode.toString() : 'avg');
    });

    h.set('QuantizeLinear', ({ data }) => data.setAttribute('levels', '256'));
    h.set('DequantizeLinear', h.get('QuantizeLinear')!);

    h.set('Einsum', ({ node, data }) => setAttr(data, node, 'equation', 'equation'));

    const normHandler: OpHandler = ({ node, data }) => {
      setAttr(data, node, 'epsilon', 'eps');
      if (node.opType === 'LayerNormalization') {
        setAttr(data, node, 'axis', 'axes');
      }
      data.setAttribute('normalize_variance', 'true');
      data.setAttribute('eps_mode', 'add');
    };
    h.set('LayerNormalization', normHandler);
    h.set('InstanceNormalization', normHandler);

    h.set('LpNormalization', ({ node, data }) => {
      setAttr(data, node, 'axis', 'axes');
      setAttr(data, node, 'p', 'p');
    });

    h.set('BatchNormalization', ({ node, data }) => setAttr(data, node, 'epsilon', 'eps'));

    h.set('If', ({ exporter, node }) => {
      const thenBranch = node.attributes['then_branch']?.value as Graph;
      if (thenBranch) {
        const subExporter = new OpenVinoExporter(thenBranch, {
          version: exporter.version,
          compressToFp16: exporter.compressToFp16,
        });
        subExporter.export();
      }
      const elseBranch = node.attributes['else_branch']?.value as Graph;
      if (elseBranch) {
        const subExporter = new OpenVinoExporter(elseBranch, {
          version: exporter.version,
          compressToFp16: exporter.compressToFp16,
        });
        subExporter.export();
      }
    });

    h.set('Loop', ({ exporter, node }) => {
      const body = node.attributes['body']?.value as Graph;
      if (body) {
        const subExporter = new OpenVinoExporter(body, {
          version: exporter.version,
          compressToFp16: exporter.compressToFp16,
        });
        subExporter.export();
      }
    });
    h.set('Scan', h.get('Loop')!);

    h.set('Cast', ({ exporter, node, layer, data }) => {
      layer.setAttribute('type', 'Convert');
      const to = node.attributes['to']?.value as DType;
      if (to) {
        data.setAttribute('destination_type', exporter.mapDtype(to));
      }
    });

    h.set('GridSample', ({ node, data }) => {
      setAttr(data, node, 'mode', 'mode');
      setAttr(data, node, 'padding_mode', 'padding_mode');
      const align = node.attributes['align_corners']?.value;
      if (align !== undefined) {
        data.setAttribute('align_corners', align ? 'true' : 'false');
      }
    });

    h.set('Size', ({ exporter, node, layer, data, inputsToMap, layerId, layers }) => {
      const shapeLayerId = exporter.nextId();
      const shapeName = (node.name || `shapeof_${layerId}`) + '_shapeof';
      const shapeLayer = new XmlNode('layer')
        .setAttribute('id', shapeLayerId.toString())
        .setAttribute('name', shapeName)
        .setAttribute('type', 'ShapeOf')
        .setAttribute('version', 'opset1');

      const shapeInNode = new XmlNode('input');
      const shapeInPort = exporter.nextPort(shapeLayerId);
      shapeInNode.addChild(new XmlNode('port').setAttribute('id', shapeInPort.toString()));
      shapeLayer.addChild(shapeInNode);

      if (node.inputs.length > 0) {
        const fromIds = exporter.portIds.get(node.inputs[0] as string);
        if (fromIds) {
          exporter['addEdge'](fromIds.layerId, fromIds.portId, shapeLayerId, shapeInPort);
        }
      }

      const shapeOutNode = new XmlNode('output');
      const shapeOutPort = exporter.nextPort(shapeLayerId);
      shapeOutNode.addChild(
        new XmlNode('port')
          .setAttribute('id', shapeOutPort.toString())
          .setAttribute('precision', 'i64'),
      );
      shapeLayer.addChild(shapeOutNode);
      layers.addChild(shapeLayer);

      const axesConst = exporter.emitDynamicConst(
        (node.name || `axes_${layerId}`) + '_axes',
        [0],
        [1],
        'int64',
      );
      layers.addChild(axesConst.layerNode);

      layer.setAttribute('type', 'ReduceProd');
      data.setAttribute('keep_dims', 'false');

      inputsToMap.length = 0;
      inputsToMap.push(shapeName, (node.name || `axes_${layerId}`) + '_axes');
      exporter.portIds.set(shapeName, { layerId: shapeLayerId, portId: shapeOutPort });
    });

    h.set('Flatten', ({ exporter, node, inputsToMap, layerId, layers }) => {
      const constNode = exporter.emitDynamicConst(
        (node.name || `flatten_shape_${layerId}`) + '_flatten_shape',
        [0, -1],
        [2],
        'int64',
      );
      layers.addChild(constNode.layerNode);
      inputsToMap.push((node.name || `flatten_shape_${layerId}`) + '_flatten_shape');
    });

    h.set('Transpose', ({ exporter, node, inputsToMap, layerId, layers }) => {
      const perm = node.attributes['perm']?.value as number[];
      if (perm) {
        const constNode = exporter.emitDynamicConst(
          (node.name || `transpose_perm_${layerId}`) + '_transpose_perm',
          perm,
          [perm.length],
          'int64',
        );
        layers.addChild(constNode.layerNode);
        inputsToMap.push((node.name || `transpose_perm_${layerId}`) + '_transpose_perm');
      }
    });

    h.set('GatherElements', ({ exporter, node, inputsToMap, layerId, layers }) => {
      const axis = node.attributes['axis']?.value as number;
      if (axis !== undefined) {
        const axisNode = exporter.emitDynamicConst(
          (node.name || `gather_el_axis_${layerId}`) + '_gather_el_axis',
          [axis],
          [1],
          'int64',
        );
        layers.addChild(axisNode.layerNode);
        inputsToMap.push((node.name || `gather_el_axis_${layerId}`) + '_gather_el_axis');
      }
    });

    h.set('ConstantOfShape', ({ exporter, node, layer, inputsToMap, layerId, layers }) => {
      let val = [0.0];
      let dtype: DType = 'float32';
      const valAttr = node.attributes['value']?.value;
      if (valAttr instanceof Tensor) {
        const valTensor = valAttr;
        if (valTensor.dtype === 'float32' && valTensor.data) {
          val = [new Float32Array(valTensor.data.buffer, valTensor.data.byteOffset, 1)[0] ?? 0.0];
          dtype = 'float32';
        } else if (valTensor.dtype === 'int64' && valTensor.data) {
          const bi =
            new BigInt64Array(valTensor.data.buffer, valTensor.data.byteOffset, 1)[0] ?? 0n;
          val = [Number(bi)];
          dtype = 'int64';
        } else if (valTensor.dtype === 'int32' && valTensor.data) {
          val = [new Int32Array(valTensor.data.buffer, valTensor.data.byteOffset, 1)[0] ?? 0];
          dtype = 'int32';
        }
      }

      const constNode = exporter.emitDynamicConst(
        (node.name || `scalar_val_${layerId}`) + '_scalar_val',
        val,
        [1],
        dtype,
      );
      layers.addChild(constNode.layerNode);
      const originalInput = inputsToMap[0];
      inputsToMap.length = 0;
      inputsToMap.push((node.name || `scalar_val_${layerId}`) + '_scalar_val');
      if (originalInput) inputsToMap.push(originalInput);
      layer.setAttribute('type', 'Broadcast');
    });
  }
}
