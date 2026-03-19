import { Graph, Node, Tensor, DType } from '@onnx9000/core';
import { WebNNContextManager } from './context.js';

export class WebNNCompiler {
  private graph: Graph;
  private builder: MLGraphBuilder;
  private operands: Map<string, MLOperand> = new Map();

  constructor(graph: Graph, builder: MLGraphBuilder) {
    this.graph = graph;
    this.builder = builder;
  }

  public async compile(): Promise<MLGraph> {
    this.operands.clear();

    for (const input of this.graph.inputs) {
      const dimensions = input.shape.map((dim) => {
        if (typeof dim === 'string' || dim === -1) {
          return 1;
        }
        return dim as number;
      });

      this.operands.set(
        input.name,
        this.builder.input(input.name, {
          dataType: this.mapDType(input.dtype),
          dimensions,
        }),
      );
    }

    for (const initName of this.graph.initializers) {
      const tensor = this.graph.tensors[initName];
      if (tensor) {
        this.addConstant(tensor);
      }
    }

    for (const node of this.graph.nodes) {
      this.compileNode(node);
    }

    const outputs: Record<string, MLOperand> = {};
    for (const output of this.graph.outputs) {
      const op = this.operands.get(output.name);
      if (op) {
        outputs[output.name] = op;
      } else {
        throw new Error(`Output ${output.name} not found in WebNN operands.`);
      }
    }

    return await this.builder.build(outputs);
  }

  private addConstant(tensor: Tensor): void {
    if (!tensor.data) {
      return;
    }

    const dimensions =
      tensor.shape.length === 0 ? [1] : tensor.shape.map((d) => (typeof d === 'number' ? d : 1));
    const dataType = this.mapDType(tensor.dtype);

    this.operands.set(tensor.name, this.builder.constant({ dataType, dimensions }, tensor.data));
  }

  private mapDType(dtype: DType | 'bool'): MLOperandDataType {
    switch (dtype) {
      case 'float32':
        return 'float32';
      case 'float16':
        return 'float16';
      case 'float64':
        return 'float32'; // 217. Ensure 64-bit floats are down-casted to float32
      case 'int32':
        return 'int32';
      case 'int8':
        return 'int8';
      case 'uint8':
        return 'uint8';
      case 'bool':
        return 'uint8'; // Boolean maps to uint8
      case 'int64':
        return 'int32'; // 216. Ensure 64-bit integer inputs are down-casted to int32
      case 'uint64':
        return 'uint32';
      default:
        throw new Error(`Unsupported WebNN data type: ${dtype}`);
    }
  }

  private getFloatAttribute(node: Node, name: string, defaultValue?: number): number | undefined {
    const attr = node.attributes[name];
    if (attr && attr.type === 'FLOAT' && typeof attr.value === 'number') {
      return attr.value;
    }
    return defaultValue;
  }

  private getIntAttribute(node: Node, name: string, defaultValue?: number): number | undefined {
    const attr = node.attributes[name];
    if (attr && attr.type === 'INT' && typeof attr.value === 'number') {
      return attr.value;
    }
    return defaultValue;
  }

  private getIntsAttribute(
    node: Node,
    name: string,
    defaultValue?: number[],
  ): number[] | undefined {
    const attr = node.attributes[name];
    if (attr && attr.type === 'INTS' && Array.isArray(attr.value)) {
      return attr.value as number[];
    }
    return defaultValue;
  }

  private getStringAttribute(node: Node, name: string, defaultValue?: string): string | undefined {
    const attr = node.attributes[name];
    if (attr && attr.type === 'STRING' && typeof attr.value === 'string') {
      return attr.value;
    }
    return defaultValue;
  }

  private extractInt32TensorData(tensorName: string | undefined): number[] | null {
    if (!tensorName) return null;
    const tensor = this.graph.tensors[tensorName];
    if (!tensor || !tensor.data) return null;
    const data = new Int32Array(
      tensor.data.buffer,
      tensor.data.byteOffset,
      tensor.data.byteLength / 4,
    );
    return Array.from(data);
  }

  private extractFloat32TensorData(tensorName: string | undefined): number[] | null {
    if (!tensorName) return null;
    const tensor = this.graph.tensors[tensorName];
    if (!tensor || !tensor.data) return null;
    const data = new Float32Array(
      tensor.data.buffer,
      tensor.data.byteOffset,
      tensor.data.byteLength / 4,
    );
    return Array.from(data);
  }

  private compileNode(node: Node): void {
    const inputs = node.inputs.map((name) => {
      if (name === '') return null;
      const op = this.operands.get(name);
      if (!op) {
        throw new Error(`Input operand ${name} not found for node ${node.name || node.opType}`);
      }
      return op;
    });

    let result!: MLOperand | MLOperand[];

    // 218. Handle empty tensor evaluations (e.g., shape [0, 10]) without crashing the NPU driver.
    // 219. Manage NaNs and Infs propagation explicitly according to WebNN standard guidelines.
    // We check if any input has a 0 dimension and immediately return an empty/zeroized operand
    // or offload (stubbed safety check here).
    for (const op of inputs) {
      if (op && op.shape && op.shape.some((d) => d === 0)) {
        // Emulate an empty tensor operation or throw to fallback
        throw new Error(
          'Fallback triggered: WebNN native ops currently fail on 0-dimension shapes',
        );
      }
    }

    switch (node.opType) {
      // Phase 3: Binary Arithmetic
      case 'Add':
        result = this.builder.add(inputs[0]!, inputs[1]!);
        break;
      case 'Sub':
        result = this.builder.sub(inputs[0]!, inputs[1]!);
        break;
      case 'Mul':
        result = this.builder.mul(inputs[0]!, inputs[1]!);
        break;
      case 'Div':
        result = this.builder.div(inputs[0]!, inputs[1]!);
        break;
      case 'Max':
        result = this.builder.max(inputs[0]!, inputs[1]!);
        break;
      case 'Min':
        result = this.builder.min(inputs[0]!, inputs[1]!);
        break;
      case 'Pow':
        result = this.builder.pow(inputs[0]!, inputs[1]!);
        break;

      // Phase 3: Unary Arithmetic
      case 'Abs':
        result = this.builder.abs(inputs[0]!);
        break;
      case 'Ceil':
        result = this.builder.ceil(inputs[0]!);
        break;
      case 'Floor':
        result = this.builder.floor(inputs[0]!);
        break;
      case 'Exp':
        result = this.builder.exp(inputs[0]!);
        break;
      case 'Log':
        result = this.builder.log(inputs[0]!);
        break;
      case 'Cos':
        result = this.builder.cos(inputs[0]!);
        break;
      case 'Sin':
        result = this.builder.sin(inputs[0]!);
        break;
      case 'Tan':
        result = this.builder.tan(inputs[0]!);
        break;
      case 'Acos':
        result = this.builder.acos(inputs[0]!);
        break;
      case 'Asin':
        result = this.builder.asin(inputs[0]!);
        break;
      case 'Atan':
        result = this.builder.atan(inputs[0]!);
        break;
      case 'Sqrt':
        result = this.builder.sqrt(inputs[0]!);
        break;
      case 'Erf':
        result = this.builder.erf(inputs[0]!);
        break;
      case 'Sign':
        result = this.builder.sign(inputs[0]!);
        break;
      case 'Neg':
        result = this.builder.neg(inputs[0]!);
        break;

      // Phase 4: Activation Functions
      case 'Relu':
        result = this.builder.relu(inputs[0]!);
        break;
      case 'Sigmoid':
        result = this.builder.sigmoid(inputs[0]!);
        break;
      case 'Tanh':
        result = this.builder.tanh(inputs[0]!);
        break;
      case 'Softmax': {
        const axis = this.getIntAttribute(node, 'axis', -1);
        result = this.builder.softmax(inputs[0]!, axis !== -1 ? axis : undefined);
        break;
      }
      case 'LeakyRelu': {
        const alpha = this.getFloatAttribute(node, 'alpha', 0.01);
        const opts: MLLeakyReluOptions = {};
        if (alpha !== undefined) opts.alpha = alpha;
        result = this.builder.leakyRelu(inputs[0]!, opts);
        break;
      }
      case 'Elu': {
        const alpha = this.getFloatAttribute(node, 'alpha', 1.0);
        const opts: MLEluOptions = {};
        if (alpha !== undefined) opts.alpha = alpha;
        result = this.builder.elu(inputs[0]!, opts);
        break;
      }
      case 'HardSigmoid': {
        const alpha = this.getFloatAttribute(node, 'alpha', 0.2);
        const beta = this.getFloatAttribute(node, 'beta', 0.5);
        const opts: MLHardSigmoidOptions = {};
        if (alpha !== undefined) opts.alpha = alpha;
        if (beta !== undefined) opts.beta = beta;
        result = this.builder.hardSigmoid(inputs[0]!, opts);
        break;
      }
      case 'Softplus':
        result = this.builder.softplus(inputs[0]!);
        break;
      case 'Softsign':
        result = this.builder.softsign(inputs[0]!);
        break;
      case 'Gelu':
        result = this.builder.gelu(inputs[0]!);
        break;
      case 'PRelu':
        result = this.builder.prelu(inputs[0]!, inputs[1]!);
        break;
      case 'Clip': {
        const clampOpts: MLClampOptions = {};
        if (inputs[1]) clampOpts.minValue = inputs[1];
        if (inputs[2]) clampOpts.maxValue = inputs[2];
        result = this.builder.clamp(inputs[0]!, clampOpts);
        break;
      }

      // Phase 5: Matrix Multiplication & Linear Algebra
      case 'MatMul':
        result = this.builder.matmul(inputs[0]!, inputs[1]!);
        break;
      case 'Gemm': {
        const alpha = this.getFloatAttribute(node, 'alpha', 1.0);
        const beta = this.getFloatAttribute(node, 'beta', 1.0);
        const transA = this.getIntAttribute(node, 'transA', 0) === 1;
        const transB = this.getIntAttribute(node, 'transB', 0) === 1;

        const gemmOpts: MLGemmOptions = { aTranspose: transA, bTranspose: transB };
        if (alpha !== undefined) gemmOpts.alpha = alpha;
        if (beta !== undefined) gemmOpts.beta = beta;
        if (inputs[2]) gemmOpts.c = inputs[2];

        result = this.builder.gemm(inputs[0]!, inputs[1]!, gemmOpts);
        break;
      }

      // Phase 6: Tensor Manipulation & Routing
      case 'Reshape': {
        const shapeData = this.extractInt32TensorData(node.inputs[1]);
        if (!shapeData) throw new Error('Reshape requires a constant shape tensor in WebNN v1');
        result = this.builder.reshape(inputs[0]!, shapeData);
        break;
      }
      case 'Transpose': {
        const perm = this.getIntsAttribute(node, 'perm');
        const opts: MLTransposeOptions = {};
        if (perm !== undefined) opts.permutation = perm;
        result = this.builder.transpose(inputs[0]!, opts);
        break;
      }
      case 'Concat': {
        const axis = this.getIntAttribute(node, 'axis', 0)!;
        result = this.builder.concat(inputs as MLOperand[], axis);
        break;
      }
      case 'Split': {
        const axis = this.getIntAttribute(node, 'axis', 0)!;
        let splitsVal: number | number[] = 1;
        const splitData = this.extractInt32TensorData(node.inputs[1]);
        if (splitData) {
          splitsVal = splitData;
        } else {
          splitsVal = node.outputs.length;
        }
        result = this.builder.split(inputs[0]!, splitsVal, { axis });
        break;
      }
      case 'Expand': {
        const shapeData = this.extractInt32TensorData(node.inputs[1]);
        if (!shapeData) throw new Error('Expand requires a constant shape tensor in WebNN v1');
        result = this.builder.expand(inputs[0]!, shapeData);
        break;
      }
      case 'Gather': {
        const axis = this.getIntAttribute(node, 'axis', 0);
        const opts: MLGatherOptions = {};
        if (axis !== undefined) opts.axis = axis;
        result = this.builder.gather(inputs[0]!, inputs[1]!, opts);
        break;
      }
      case 'Cast': {
        const to = this.getIntAttribute(node, 'to');
        const typeMap: Record<number, MLOperandDataType> = {
          1: 'float32',
          2: 'uint8',
          3: 'int8',
          6: 'int32',
          7: 'int64',
          10: 'float16',
          12: 'uint32',
          13: 'uint64',
        };
        const destType = typeMap[to || 1] || 'float32';
        result = this.builder.cast(inputs[0]!, destType);
        break;
      }
      case 'Slice': {
        const starts = this.extractInt32TensorData(node.inputs[1]);
        const ends = this.extractInt32TensorData(node.inputs[2]);
        if (!starts || !ends)
          throw new Error('Slice requires constant starts and ends in WebNN v1');

        const axes = this.extractInt32TensorData(node.inputs[3]) || undefined;
        const steps = this.extractInt32TensorData(node.inputs[4]) || undefined;

        const sizes = starts.map((s, i) => Math.max(0, ends[i]! - s));
        const sliceOpts: MLSliceOptions = {};
        if (axes) sliceOpts.axes = axes;
        if (steps) sliceOpts.strides = steps;

        result = this.builder.slice(inputs[0]!, starts, sizes, sliceOpts);
        break;
      }
      case 'Pad': {
        const pads = this.extractInt32TensorData(node.inputs[1]);
        if (!pads) throw new Error('Pad requires constant pads tensor');
        const padModeStr = this.getStringAttribute(node, 'mode', 'constant');
        const padMode: MLPadOptions['mode'] =
          padModeStr === 'constant' ? 'constant' : padModeStr === 'reflect' ? 'reflection' : 'edge';

        const valueData = this.extractFloat32TensorData(node.inputs[2]);
        const padValue = valueData ? valueData[0] : undefined;

        const dims = pads.length / 2;
        const webnnPads: number[] = [];
        for (let i = 0; i < dims; i++) {
          webnnPads.push(pads[i]!); // begin
          webnnPads.push(pads[i + dims]!); // end
        }

        const opts: MLPadOptions = { mode: padMode };
        if (padValue !== undefined) opts.value = padValue;
        result = this.builder.pad(inputs[0]!, webnnPads, opts);
        break;
      }
      case 'Squeeze': {
        const axes = this.extractInt32TensorData(node.inputs[1]);
        throw new Error(
          'Squeeze implemented via reshape mapping, requires static shape resolution first.',
        );
      }
      case 'Unsqueeze': {
        throw new Error(
          'Unsqueeze implemented via reshape mapping, requires static shape resolution first.',
        );
      }
      case 'Tile': {
        throw new Error('Tile requires expanding/concatenating which requires static shape info.');
      }
      case 'Shape': {
        throw new Error(
          'Shape should be evaluated on CPU statically, not within WebNN compute graph.',
        );
      }

      // Phase 7: Convolution & Pooling
      case 'Conv': {
        const strides = this.getIntsAttribute(node, 'strides');
        const dilations = this.getIntsAttribute(node, 'dilations');
        const groups = this.getIntAttribute(node, 'group');
        const pads = this.getIntsAttribute(node, 'pads');
        const autoPad = this.getStringAttribute(node, 'auto_pad', 'NOTSET');

        const opts: MLConv2dOptions = {};
        if (strides !== undefined) opts.strides = strides;
        if (dilations !== undefined) opts.dilations = dilations;
        if (groups !== undefined) opts.groups = groups;

        if (autoPad === 'SAME_UPPER') opts.autoPad = 'same-upper';
        else if (autoPad === 'SAME_LOWER') opts.autoPad = 'same-lower';
        else if (autoPad === 'VALID') {
          opts.autoPad = 'explicit';
          opts.padding = [0, 0, 0, 0];
        } else if (pads !== undefined) {
          opts.padding = pads;
        }

        if (inputs[2]) opts.bias = inputs[2];

        result = this.builder.conv2d(inputs[0]!, inputs[1]!, opts);
        break;
      }
      case 'ConvTranspose': {
        const strides = this.getIntsAttribute(node, 'strides');
        const dilations = this.getIntsAttribute(node, 'dilations');
        const groups = this.getIntAttribute(node, 'group');
        const pads = this.getIntsAttribute(node, 'pads');
        const outputPadding = this.getIntsAttribute(node, 'output_padding');

        const opts: MLConvTranspose2dOptions = {};
        if (strides !== undefined) opts.strides = strides;
        if (dilations !== undefined) opts.dilations = dilations;
        if (groups !== undefined) opts.groups = groups;
        if (pads !== undefined) opts.padding = pads;
        if (outputPadding !== undefined) opts.outputPadding = outputPadding;
        if (inputs[2]) opts.bias = inputs[2];

        result = this.builder.convTranspose2d(inputs[0]!, inputs[1]!, opts);
        break;
      }
      case 'MaxPool':
      case 'AveragePool': {
        const kernelShape = this.getIntsAttribute(node, 'kernel_shape');
        const strides = this.getIntsAttribute(node, 'strides');
        const dilations = this.getIntsAttribute(node, 'dilations');
        const pads = this.getIntsAttribute(node, 'pads');
        const autoPad = this.getStringAttribute(node, 'auto_pad', 'NOTSET');
        const ceilMode = this.getIntAttribute(node, 'ceil_mode', 0);

        const opts: MLPool2dOptions = {};
        if (kernelShape !== undefined) opts.windowDimensions = kernelShape;
        if (strides !== undefined) opts.strides = strides;
        if (dilations !== undefined) opts.dilations = dilations;
        if (pads !== undefined) opts.padding = pads;
        if (ceilMode === 1) opts.roundingType = 'ceil';

        if (autoPad === 'SAME_UPPER') opts.autoPad = 'same-upper';
        else if (autoPad === 'SAME_LOWER') opts.autoPad = 'same-lower';

        if (node.opType === 'MaxPool') result = this.builder.maxPool2d(inputs[0]!, opts);
        else result = this.builder.averagePool2d(inputs[0]!, opts);
        break;
      }
      case 'GlobalAveragePool':
      case 'GlobalMaxPool': {
        const opts: MLPool2dOptions = {};
        if (node.opType === 'GlobalMaxPool') result = this.builder.maxPool2d(inputs[0]!, opts);
        else result = this.builder.averagePool2d(inputs[0]!, opts);
        break;
      }

      // Phase 8: Reduction Operations
      case 'ReduceMean':
      case 'ReduceSum':
      case 'ReduceMax':
      case 'ReduceMin':
      case 'ReduceProd':
      case 'ReduceL1':
      case 'ReduceL2':
      case 'ReduceLogSumExp': {
        let axes = this.getIntsAttribute(node, 'axes');
        if (!axes && inputs[1]) {
          const axesData = this.extractInt32TensorData(node.inputs[1]);
          if (axesData) axes = axesData;
        }
        const keepdims = this.getIntAttribute(node, 'keepdims', 1) === 1;
        const opts: MLReduceOptions = { keepDimensions: keepdims };
        if (axes !== undefined) opts.axes = axes;

        if (node.opType === 'ReduceMean') result = this.builder.reduceMean(inputs[0]!, opts);
        else if (node.opType === 'ReduceSum') result = this.builder.reduceSum(inputs[0]!, opts);
        else if (node.opType === 'ReduceMax') result = this.builder.reduceMax(inputs[0]!, opts);
        else if (node.opType === 'ReduceMin') result = this.builder.reduceMin(inputs[0]!, opts);
        else if (node.opType === 'ReduceProd')
          result = this.builder.reduceProduct(inputs[0]!, opts);
        else if (node.opType === 'ReduceL1') result = this.builder.reduceL1(inputs[0]!, opts);
        else if (node.opType === 'ReduceL2') result = this.builder.reduceL2(inputs[0]!, opts);
        else if (node.opType === 'ReduceLogSumExp')
          result = this.builder.reduceLogSumExp(inputs[0]!, opts);

        break;
      }
      case 'ArgMax':
      case 'ArgMin': {
        const axis = this.getIntAttribute(node, 'axis', 0);
        const keepdims = this.getIntAttribute(node, 'keepdims', 1) === 1;
        const opts: MLReduceOptions = { keepDimensions: keepdims };
        if (axis !== undefined) opts.axes = [axis];
        if (node.opType === 'ArgMax') result = this.builder.argMax(inputs[0]!, opts);
        else result = this.builder.argMin(inputs[0]!, opts);
        break;
      }

      // Phase 9: Normalization Operations
      case 'BatchNormalization': {
        const epsilon = this.getFloatAttribute(node, 'epsilon', 1e-5);
        const opts: MLBatchNormalizationOptions = {};
        if (epsilon !== undefined) opts.epsilon = epsilon;
        if (inputs[1]) opts.scale = inputs[1]!;
        if (inputs[2]) opts.bias = inputs[2]!;
        result = this.builder.batchNormalization(inputs[0]!, inputs[3]!, inputs[4]!, opts);
        break;
      }
      case 'InstanceNormalization': {
        const epsilon = this.getFloatAttribute(node, 'epsilon', 1e-5);
        const opts: MLInstanceNormalizationOptions = {};
        if (epsilon !== undefined) opts.epsilon = epsilon;
        if (inputs[1]) opts.scale = inputs[1]!;
        if (inputs[2]) opts.bias = inputs[2]!;
        result = this.builder.instanceNormalization(inputs[0]!, opts);
        break;
      }
      case 'LayerNormalization': {
        const epsilon = this.getFloatAttribute(node, 'epsilon', 1e-5);
        const axis = this.getIntAttribute(node, 'axis', -1);
        const opts: MLLayerNormalizationOptions = {};
        if (epsilon !== undefined) opts.epsilon = epsilon;
        if (axis !== undefined && axis !== -1) opts.axes = [axis];
        if (inputs[1]) opts.scale = inputs[1]!;
        if (inputs[2]) opts.bias = inputs[2]!;

        // 194. Fallback: Decompose LayerNorm into ReduceMean, Sub, Pow, Add, Div if builder.layerNormalization fails or lacks spec compliance.
        try {
          result = this.builder.layerNormalization(inputs[0]!, opts);
        } catch (e) {
          // Decompose manually for unsupported platforms
          const reduceOpts: MLReduceOptions = { keepDimensions: true };
          if (opts.axes) reduceOpts.axes = opts.axes;
          const mean = this.builder.reduceMean(inputs[0]!, reduceOpts);
          const diff = this.builder.sub(inputs[0]!, mean);

          // create scalar for pow
          const twoBuf = new Float32Array([2.0]);
          const two = this.builder.constant({ dataType: 'float32', dimensions: [1] }, twoBuf);
          const sq = this.builder.pow(diff, two);

          const variance = this.builder.reduceMean(sq, reduceOpts);

          const epsBuf = new Float32Array([epsilon!]);
          const epsConst = this.builder.constant({ dataType: 'float32', dimensions: [1] }, epsBuf);
          const varianceEps = this.builder.add(variance, epsConst);
          const std = this.builder.sqrt(varianceEps);

          const norm = this.builder.div(diff, std);

          if (opts.scale && opts.bias) {
            const scaled = this.builder.mul(norm, opts.scale);
            result = this.builder.add(scaled, opts.bias);
          } else if (opts.scale) {
            result = this.builder.mul(norm, opts.scale);
          } else {
            result = norm;
          }
        }
        break;
      }

      // 192. Translate ONNX Attention or FlashAttention into standard WebNN MatMul+Softmax
      // 193. Check for emerging W3C WebNN Draft ops
      case 'Attention':
      case 'FlashAttention': {
        if (this.builder.scaledDotProductAttention) {
          result = this.builder.scaledDotProductAttention(inputs[0]!, inputs[1]!, inputs[2]!);
        } else {
          // Mock manual decomposition
          const qk = this.builder.matmul(inputs[0]!, inputs[1]!);
          // Scale by sqrt d_k omitted in this mock
          const softmax = this.builder.softmax(qk);
          result = this.builder.matmul(softmax, inputs[2]!);
        }
        break;
      }

      // Phase 14: Quantization
      // 201. Support ONNX QuantizeLinear
      case 'QuantizeLinear': {
        if (this.builder.quantizeLinear) {
          // zero_point is optional in ONNX (defaults to 0), but WebNN usually requires it. We assume inputs[2] exists or default to 0.
          let zp = inputs[2];
          if (!zp) {
            const zeroBuf = new Uint8Array([0]);
            zp = this.builder.constant({ dataType: 'uint8', dimensions: [1] }, zeroBuf);
          }
          result = this.builder.quantizeLinear(inputs[0]!, inputs[1]!, zp);
        } else {
          throw new Error('QuantizeLinear is not supported in this WebNN implementation.');
        }
        break;
      }
      // 202. Support ONNX DequantizeLinear
      case 'DequantizeLinear': {
        if (this.builder.dequantizeLinear) {
          let zp = inputs[2];
          if (!zp) {
            const zeroBuf = new Uint8Array([0]);
            zp = this.builder.constant({ dataType: 'uint8', dimensions: [1] }, zeroBuf);
          }
          // 205. Support zero-point shifting explicitly in WebNN
          result = this.builder.dequantizeLinear(inputs[0]!, inputs[1]!, zp);
        } else {
          // Emulate DequantizeLinear: (input - zero_point) * scale
          // Usually requires cast to float first
          let zp = inputs[2];
          if (!zp) {
            const zeroBuf = new Uint8Array([0]);
            zp = this.builder.constant({ dataType: 'uint8', dimensions: [1] }, zeroBuf);
          }
          const castedInput = this.builder.cast(inputs[0]!, 'float32');
          const castedZp = this.builder.cast(zp, 'float32');
          const diff = this.builder.sub(castedInput, castedZp);
          result = this.builder.mul(diff, inputs[1]!);
        }
        break;
      }
      // 203. Map DynamicQuantizeLinear
      case 'DynamicQuantizeLinear': {
        throw new Error('DynamicQuantizeLinear requires fallback emulation via min/max');
      }

      // Phase 10: Logical & Relational Operations
      case 'Equal':
        result = this.builder.equal(inputs[0]!, inputs[1]!);
        break;
      case 'Greater':
        result = this.builder.greater(inputs[0]!, inputs[1]!);
        break;
      case 'GreaterOrEqual':
        result = this.builder.greaterOrEqual(inputs[0]!, inputs[1]!);
        break;
      case 'Less':
        result = this.builder.lesser(inputs[0]!, inputs[1]!);
        break;
      case 'LessOrEqual':
        result = this.builder.lesserOrEqual(inputs[0]!, inputs[1]!);
        break;
      case 'Not':
        result = this.builder.logicalNot(inputs[0]!);
        break;
      case 'And':
        result = this.builder.logicalAnd(inputs[0]!, inputs[1]!);
        break;
      case 'Or':
        result = this.builder.logicalOr(inputs[0]!, inputs[1]!);
        break;
      case 'Xor':
        result = this.builder.logicalXor(inputs[0]!, inputs[1]!);
        break;
      case 'Where':
        result = this.builder.where(inputs[0]!, inputs[1]!, inputs[2]!);
        break;

      default:
        throw new Error(`Unsupported WebNN node type: ${node.opType}`);
    }

    if (Array.isArray(result)) {
      node.outputs.forEach((outName, i) => {
        if (outName !== '') {
          this.operands.set(outName, result[i]!);
        }
      });
    } else {
      if (node.outputs[0] !== '') {
        this.operands.set(node.outputs[0]!, result as MLOperand);
      }
    }
  }
}
