import {
  MLContext,
  MLOperandDescriptor,
  MLOperand,
  MLGraph,
  MLClampOptions,
  MLLeakyReluOptions,
  MLEluOptions,
  MLHardSigmoidOptions,
  MLGemmOptions,
  MLTransposeOptions,
  MLSliceOptions,
  MLPadOptions,
  MLGatherOptions,
  MLSplitOptions,
  MLConv2dOptions,
  MLConvTranspose2dOptions,
  MLPool2dOptions,
  MLReduceOptions,
  MLBatchNormalizationOptions,
  MLInstanceNormalizationOptions,
  MLLayerNormalizationOptions,
  MLOperandDataType,
} from './interfaces.js';
import { PolyfillMLOperand } from './operand.js';
import { PolyfillMLGraph } from './graph.js';
import { PolyfillMLContext } from './context.js';
import { Graph, Node, Tensor, ValueInfo, Attribute } from '@onnx9000/core';

export class PolyfillMLGraphBuilder {
  private context: PolyfillMLContext;
  private graph: Graph;
  private nodeCounter: number = 0;

  constructor(context: MLContext) {
    this.context = context as PolyfillMLContext;
    this.graph = new Graph('WebNN_Polyfill_Graph');
  }

  private nextName(op: string): string {
    return `${op}_${this.nodeCounter++}`;
  }

  // 18. Support builder.input(name, descriptor).
  input(name: string, descriptor: MLOperandDescriptor): MLOperand {
    // 19. Validate MLOperandDescriptor shapes strictly (Array of positive integers).
    for (const d of descriptor.dimensions) {
      if (d <= 0 || !Number.isInteger(d)) {
        throw new DOMException(`Invalid dimension ${d} in shape`, 'DataError');
      }
    }
    // 20. Validate MLOperandDescriptor datatypes strictly
    const validTypes = [
      'float32',
      'float16',
      'int32',
      'uint32',
      'int8',
      'uint8',
      'int64',
      'uint64',
    ];
    if (!validTypes.includes(descriptor.dataType)) {
      throw new DOMException(`Invalid dataType ${descriptor.dataType}`, 'DataError');
    }

    this.graph.inputs.push(new ValueInfo(name, descriptor.dimensions, descriptor.dataType as any));
    return new PolyfillMLOperand(name, descriptor.dataType, descriptor.dimensions);
  }

  // 22. Support builder.constant(descriptor, bufferView).
  constant(descriptor: MLOperandDescriptor, bufferView: ArrayBufferView): MLOperand {
    const name = this.nextName('Constant');
    // 23. Extract ArrayBuffer values from constant() calls into onnx9000 Initializers natively.
    const tensor = new Tensor(
      name,
      descriptor.dimensions,
      descriptor.dataType as any,
      true,
      false,
      bufferView,
    );
    this.graph.addTensor(tensor);
    this.graph.initializers.push(name);
    return new PolyfillMLOperand(name, descriptor.dataType, descriptor.dimensions);
  }

  async build(outputs: Record<string, MLOperand>): Promise<MLGraph> {
    for (const [key, op] of Object.entries(outputs)) {
      const polyOp = op as PolyfillMLOperand;
      if (polyOp.name !== key) {
        // Emit an identity node to map the name to the expected output key
        const node = new Node('Identity', [polyOp.name], [key]);
        this.graph.addNode(node);
        this.graph.outputs.push(new ValueInfo(key, polyOp.shape, polyOp.dataType as any));
      } else {
        this.graph.outputs.push(new ValueInfo(polyOp.name, polyOp.shape, polyOp.dataType as any));
      }
    }
    // 151. Execute onnx9000.shape_inference natively...
    // 152. Execute onnx9000.optimizer natively...
    // Return PolyfillMLGraph (150)
    return new PolyfillMLGraph(this.graph);
  }

  private emitNode(
    opType: string,
    inputs: MLOperand[],
    attributes: Record<string, Attribute> = {},
  ): MLOperand {
    const name = this.nextName(opType);
    const inNames = inputs.map((i) => (i as PolyfillMLOperand).name);
    const outName = `${name}_out`;

    // Simplistic shape/type inference for the polyfill (we rely on ONNX standard later, but provide a basic shape)
    const outType = inputs[0]?.dataType || 'float32';
    const outShape = inputs[0]?.shape || []; // Broadcasting not strictly implemented here, just passing through

    const node = new Node(opType, inNames, [outName], attributes, name);
    this.graph.addNode(node);

    return new PolyfillMLOperand(outName, outType, outShape);
  }

  // Phase 3: Element-wise Binary & Unary Operations
  add(a: MLOperand, b: MLOperand): MLOperand {
    return this.emitNode('Add', [a, b]);
  }
  sub(a: MLOperand, b: MLOperand): MLOperand {
    return this.emitNode('Sub', [a, b]);
  }
  mul(a: MLOperand, b: MLOperand): MLOperand {
    return this.emitNode('Mul', [a, b]);
  }
  div(a: MLOperand, b: MLOperand): MLOperand {
    return this.emitNode('Div', [a, b]);
  }
  max(a: MLOperand, b: MLOperand): MLOperand {
    return this.emitNode('Max', [a, b]);
  }
  min(a: MLOperand, b: MLOperand): MLOperand {
    return this.emitNode('Min', [a, b]);
  }
  pow(a: MLOperand, b: MLOperand): MLOperand {
    return this.emitNode('Pow', [a, b]);
  }
  abs(a: MLOperand): MLOperand {
    return this.emitNode('Abs', [a]);
  }
  ceil(a: MLOperand): MLOperand {
    return this.emitNode('Ceil', [a]);
  }
  floor(a: MLOperand): MLOperand {
    return this.emitNode('Floor', [a]);
  }
  exp(a: MLOperand): MLOperand {
    return this.emitNode('Exp', [a]);
  }
  log(a: MLOperand): MLOperand {
    return this.emitNode('Log', [a]);
  }
  cos(a: MLOperand): MLOperand {
    return this.emitNode('Cos', [a]);
  }
  sin(a: MLOperand): MLOperand {
    return this.emitNode('Sin', [a]);
  }
  tan(a: MLOperand): MLOperand {
    return this.emitNode('Tan', [a]);
  }
  acos(a: MLOperand): MLOperand {
    return this.emitNode('Acos', [a]);
  }
  asin(a: MLOperand): MLOperand {
    return this.emitNode('Asin', [a]);
  }
  atan(a: MLOperand): MLOperand {
    return this.emitNode('Atan', [a]);
  }
  sqrt(a: MLOperand): MLOperand {
    return this.emitNode('Sqrt', [a]);
  }
  erf(a: MLOperand): MLOperand {
    return this.emitNode('Erf', [a]);
  }
  sign(a: MLOperand): MLOperand {
    return this.emitNode('Sign', [a]);
  }
  neg(a: MLOperand): MLOperand {
    return this.emitNode('Neg', [a]);
  }

  // Phase 4: Activations & Non-Linearities
  relu(a: MLOperand): MLOperand {
    return this.emitNode('Relu', [a]);
  }
  sigmoid(a: MLOperand): MLOperand {
    return this.emitNode('Sigmoid', [a]);
  }
  tanh(a: MLOperand): MLOperand {
    return this.emitNode('Tanh', [a]);
  }
  softmax(a: MLOperand, axis?: number): MLOperand {
    return this.emitNode(
      'Softmax',
      [a],
      axis !== undefined ? { axis: new Attribute('axis', 'INT', axis) } : {},
    );
  }
  leakyRelu(a: MLOperand, options?: MLLeakyReluOptions): MLOperand {
    return this.emitNode(
      'LeakyRelu',
      [a],
      options?.alpha !== undefined ? { alpha: new Attribute('alpha', 'FLOAT', options.alpha) } : {},
    );
  }
  elu(a: MLOperand, options?: MLEluOptions): MLOperand {
    return this.emitNode(
      'Elu',
      [a],
      options?.alpha !== undefined ? { alpha: new Attribute('alpha', 'FLOAT', options.alpha) } : {},
    );
  }
  hardSigmoid(a: MLOperand, options?: MLHardSigmoidOptions): MLOperand {
    const attrs: Record<string, Attribute> = {};
    if (options?.alpha !== undefined) attrs.alpha = new Attribute('alpha', 'FLOAT', options.alpha);
    if (options?.beta !== undefined) attrs.beta = new Attribute('beta', 'FLOAT', options.beta);
    return this.emitNode('HardSigmoid', [a], attrs);
  }
  hardSwish(a: MLOperand): MLOperand {
    return this.emitNode('HardSwish', [a]);
  }
  linear(a: MLOperand, options?: any): MLOperand {
    const alpha = options?.alpha !== undefined ? options.alpha : 1.0;
    const beta = options?.beta !== undefined ? options.beta : 0.0;
    const alphaOp = this.constant(
      { dataType: a.dataType, dimensions: [] },
      new Float32Array([alpha]),
    );
    const betaOp = this.constant(
      { dataType: a.dataType, dimensions: [] },
      new Float32Array([beta]),
    );
    const mulOp = this.emitNode('Mul', [a, alphaOp]);
    return this.emitNode('Add', [mulOp, betaOp]);
  }

  softplus(a: MLOperand): MLOperand {
    return this.emitNode('Softplus', [a]);
  }
  softsign(a: MLOperand): MLOperand {
    return this.emitNode('Softsign', [a]);
  }
  gelu(a: MLOperand): MLOperand {
    return this.emitNode('Gelu', [a]);
  }
  prelu(a: MLOperand, slope: MLOperand): MLOperand {
    return this.emitNode('PRelu', [a, slope]);
  }
  clamp(a: MLOperand, options?: MLClampOptions): MLOperand {
    const inputs = [a];
    if (options?.minValue) inputs.push(options.minValue);
    if (options?.maxValue) {
      if (!options.minValue)
        inputs.push(
          this.constant({ dataType: a.dataType, dimensions: [] }, new Float32Array([-Infinity])),
        );
      inputs.push(options.maxValue);
    }
    return this.emitNode('Clip', inputs);
  }

  // Phase 4: Matrix Multiplication & Linear Algebra
  matmul(a: MLOperand, b: MLOperand): MLOperand {
    return this.emitNode('MatMul', [a, b]);
  }
  gemm(a: MLOperand, b: MLOperand, options?: MLGemmOptions): MLOperand {
    const attrs: Record<string, Attribute> = {};
    if (options?.alpha !== undefined) attrs.alpha = new Attribute('alpha', 'FLOAT', options.alpha);
    if (options?.beta !== undefined) attrs.beta = new Attribute('beta', 'FLOAT', options.beta);
    if (options?.aTranspose) attrs.transA = new Attribute('transA', 'INT', 1);
    if (options?.bTranspose) attrs.transB = new Attribute('transB', 'INT', 1);
    const inputs = [a, b];
    if (options?.c) inputs.push(options.c);
    return this.emitNode('Gemm', inputs, attrs);
  }

  // Phase 8: Tensor Manipulation
  reshape(a: MLOperand, newShape: number[]): MLOperand {
    const shapeOp = this.constant(
      { dataType: 'int64', dimensions: [newShape.length] },
      new BigInt64Array(newShape.map(BigInt)),
    );
    return this.emitNode('Reshape', [a, shapeOp]);
  }
  transpose(a: MLOperand, options?: MLTransposeOptions): MLOperand {
    return this.emitNode(
      'Transpose',
      [a],
      options?.permutation ? { perm: new Attribute('perm', 'INTS', options.permutation) } : {},
    );
  }
  slice(a: MLOperand, starts: number[], sizes: number[], options?: MLSliceOptions): MLOperand {
    const ends = starts.map((s, i) => s + sizes[i]!);
    const startsOp = this.constant(
      { dataType: 'int64', dimensions: [starts.length] },
      new BigInt64Array(starts.map(BigInt)),
    );
    const endsOp = this.constant(
      { dataType: 'int64', dimensions: [ends.length] },
      new BigInt64Array(ends.map(BigInt)),
    );
    const inputs = [a, startsOp, endsOp];
    if (options?.axes) {
      inputs.push(
        this.constant(
          { dataType: 'int64', dimensions: [options.axes.length] },
          new BigInt64Array(options.axes.map(BigInt)),
        ),
      );
    }
    if (options?.strides) {
      if (!options.axes)
        inputs.push(
          this.constant(
            { dataType: 'int64', dimensions: [starts.length] },
            new BigInt64Array(starts.map((_, i) => BigInt(i))),
          ),
        );
      inputs.push(
        this.constant(
          { dataType: 'int64', dimensions: [options.strides.length] },
          new BigInt64Array(options.strides.map(BigInt)),
        ),
      );
    }
    return this.emitNode('Slice', inputs);
  }
  concat(inputs: MLOperand[], axis: number): MLOperand {
    return this.emitNode('Concat', inputs, { axis: new Attribute('axis', 'INT', axis) });
  }
  split(a: MLOperand, splits: number | number[], options?: MLSplitOptions): MLOperand[] {
    const attrs: Record<string, Attribute> = {};
    if (options?.axis !== undefined) attrs.axis = new Attribute('axis', 'INT', options.axis);

    // Simplistic return of multiple operands
    const name = this.nextName('Split');
    const inNames = [(a as PolyfillMLOperand).name];
    let numOutputs = typeof splits === 'number' ? splits : splits.length;
    const outNames = Array.from({ length: numOutputs }, (_, i) => `${name}_out${i}`);

    let inputs = [a];
    if (Array.isArray(splits)) {
      inputs.push(
        this.constant(
          { dataType: 'int64', dimensions: [splits.length] },
          new BigInt64Array(splits.map(BigInt)),
        ),
      );
    }

    const node = new Node(
      'Split',
      inputs.map((i) => (i as PolyfillMLOperand).name),
      outNames,
      attrs,
      name,
    );
    this.graph.addNode(node);
    return outNames.map((o) => new PolyfillMLOperand(o, a.dataType, []));
  }
  expand(a: MLOperand, newShape: number[]): MLOperand {
    const shapeOp = this.constant(
      { dataType: 'int64', dimensions: [newShape.length] },
      new BigInt64Array(newShape.map(BigInt)),
    );
    return this.emitNode('Expand', [a, shapeOp]);
  }
  gather(a: MLOperand, indices: MLOperand, options?: MLGatherOptions): MLOperand {
    return this.emitNode(
      'Gather',
      [a, indices],
      options?.axis !== undefined ? { axis: new Attribute('axis', 'INT', options.axis) } : {},
    );
  }
  gatherNd(a: MLOperand, indices: MLOperand): MLOperand {
    return this.emitNode('GatherND', [a, indices]);
  }
  scatterNd(indices: MLOperand, updates: MLOperand, options?: any): MLOperand {
    return this.emitNode('ScatterND', [
      this.constant(
        { dataType: updates.dataType, dimensions: options?.shape || [] },
        new Float32Array(
          options?.shape ? options.shape.reduce((a: number, b: number) => a * b, 1) : 0,
        ),
      ),
      indices,
      updates,
    ]);
  }
  gatherElements(a: MLOperand, indices: MLOperand, options?: any): MLOperand {
    return this.emitNode(
      'GatherElements',
      [a, indices],
      options?.axis !== undefined ? { axis: new Attribute('axis', 'INT', options.axis) } : {},
    );
  }

  pad(a: MLOperand, padding: number[], options?: MLPadOptions): MLOperand {
    const attrs: Record<string, Attribute> = {};
    if (options?.mode) attrs.mode = new Attribute('mode', 'STRING', options.mode);
    const padsOp = this.constant(
      { dataType: 'int64', dimensions: [padding.length] },
      new BigInt64Array(padding.map(BigInt)),
    );
    const inputs = [a, padsOp];
    if (options?.value !== undefined) {
      inputs.push(
        this.constant({ dataType: a.dataType, dimensions: [] }, new Float32Array([options.value])),
      );
    }
    return this.emitNode('Pad', inputs, attrs);
  }
  cast(a: MLOperand, type: MLOperandDataType): MLOperand {
    // Map webnn types to ONNX tensorproto types
    const typeMap: Record<string, number> = {
      float32: 1,
      uint8: 2,
      int8: 3,
      uint16: 4,
      int16: 5,
      int32: 6,
      int64: 7,
      string: 8,
      bool: 9,
      float16: 10,
      float64: 11,
      uint32: 12,
      uint64: 13,
    };
    return this.emitNode('Cast', [a], { to: new Attribute('to', 'INT', typeMap[type] || 1) });
  }

  // Phase 5: Convolution & Pooling
  conv2d(input: MLOperand, filter: MLOperand, options?: MLConv2dOptions): MLOperand {
    const attrs: Record<string, Attribute> = {};
    if (options?.padding) attrs.pads = new Attribute('pads', 'INTS', options.padding);
    if (options?.strides) attrs.strides = new Attribute('strides', 'INTS', options.strides);
    if (options?.dilations) attrs.dilations = new Attribute('dilations', 'INTS', options.dilations);
    if (options?.groups) attrs.group = new Attribute('group', 'INT', options.groups);
    if (options?.autoPad) {
      const padMap: Record<string, string> = {
        explicit: 'NOTSET',
        'same-upper': 'SAME_UPPER',
        'same-lower': 'SAME_LOWER',
      };
      attrs.auto_pad = new Attribute('auto_pad', 'STRING', padMap[options.autoPad] || 'NOTSET');
    }
    const inputs = [input, filter];
    if (options?.bias) inputs.push(options.bias);
    return this.emitNode('Conv', inputs, attrs);
  }
  convTranspose2d(
    input: MLOperand,
    filter: MLOperand,
    options?: MLConvTranspose2dOptions,
  ): MLOperand {
    const attrs: Record<string, Attribute> = {};
    if (options?.padding) attrs.pads = new Attribute('pads', 'INTS', options.padding);
    if (options?.strides) attrs.strides = new Attribute('strides', 'INTS', options.strides);
    if (options?.dilations) attrs.dilations = new Attribute('dilations', 'INTS', options.dilations);
    if (options?.groups) attrs.group = new Attribute('group', 'INT', options.groups);
    if (options?.outputPadding)
      attrs.output_padding = new Attribute('output_padding', 'INTS', options.outputPadding);
    const inputs = [input, filter];
    if (options?.bias) inputs.push(options.bias);
    return this.emitNode('ConvTranspose', inputs, attrs);
  }
  maxPool2d(input: MLOperand, options?: MLPool2dOptions): MLOperand {
    const attrs: Record<string, Attribute> = {};
    if (options?.windowDimensions)
      attrs.kernel_shape = new Attribute('kernel_shape', 'INTS', options.windowDimensions);
    if (options?.padding) attrs.pads = new Attribute('pads', 'INTS', options.padding);
    if (options?.strides) attrs.strides = new Attribute('strides', 'INTS', options.strides);
    if (options?.dilations) attrs.dilations = new Attribute('dilations', 'INTS', options.dilations);
    if (options?.roundingType === 'ceil') attrs.ceil_mode = new Attribute('ceil_mode', 'INT', 1);
    return this.emitNode('MaxPool', [input], attrs);
  }
  averagePool2d(input: MLOperand, options?: MLPool2dOptions): MLOperand {
    const attrs: Record<string, Attribute> = {};
    if (options?.windowDimensions)
      attrs.kernel_shape = new Attribute('kernel_shape', 'INTS', options.windowDimensions);
    if (options?.padding) attrs.pads = new Attribute('pads', 'INTS', options.padding);
    if (options?.strides) attrs.strides = new Attribute('strides', 'INTS', options.strides);
    if (options?.roundingType === 'ceil') attrs.ceil_mode = new Attribute('ceil_mode', 'INT', 1);
    return this.emitNode('AveragePool', [input], attrs);
  }
  l2Pool2d(input: MLOperand, options?: MLPool2dOptions): MLOperand {
    const attrs: Record<string, Attribute> = {};
    if (options?.windowDimensions)
      attrs.kernel_shape = new Attribute('kernel_shape', 'INTS', options.windowDimensions);
    if (options?.padding) attrs.pads = new Attribute('pads', 'INTS', options.padding);
    if (options?.strides) attrs.strides = new Attribute('strides', 'INTS', options.strides);
    return this.emitNode('LpPool', [input], attrs); // p=2 default
  }

  // Phase 9: Reduction
  reduceMean(input: MLOperand, options?: MLReduceOptions): MLOperand {
    return this.emitNode('ReduceMean', [input], {
      axes: new Attribute('axes', 'INTS', options?.axes || []),
      keepdims: new Attribute('keepdims', 'INT', options?.keepDimensions === false ? 0 : 1),
    });
  }
  reduceSum(input: MLOperand, options?: MLReduceOptions): MLOperand {
    return this.emitNode('ReduceSum', [input], {
      axes: new Attribute('axes', 'INTS', options?.axes || []),
      keepdims: new Attribute('keepdims', 'INT', options?.keepDimensions === false ? 0 : 1),
    });
  }
  reduceMax(input: MLOperand, options?: MLReduceOptions): MLOperand {
    return this.emitNode('ReduceMax', [input], {
      axes: new Attribute('axes', 'INTS', options?.axes || []),
      keepdims: new Attribute('keepdims', 'INT', options?.keepDimensions === false ? 0 : 1),
    });
  }
  reduceMin(input: MLOperand, options?: MLReduceOptions): MLOperand {
    return this.emitNode('ReduceMin', [input], {
      axes: new Attribute('axes', 'INTS', options?.axes || []),
      keepdims: new Attribute('keepdims', 'INT', options?.keepDimensions === false ? 0 : 1),
    });
  }
  reduceProduct(input: MLOperand, options?: MLReduceOptions): MLOperand {
    return this.emitNode('ReduceProd', [input], {
      axes: new Attribute('axes', 'INTS', options?.axes || []),
      keepdims: new Attribute('keepdims', 'INT', options?.keepDimensions === false ? 0 : 1),
    });
  }
  reduceL1(input: MLOperand, options?: MLReduceOptions): MLOperand {
    return this.emitNode('ReduceL1', [input], {
      axes: new Attribute('axes', 'INTS', options?.axes || []),
      keepdims: new Attribute('keepdims', 'INT', options?.keepDimensions === false ? 0 : 1),
    });
  }
  reduceL2(input: MLOperand, options?: MLReduceOptions): MLOperand {
    return this.emitNode('ReduceL2', [input], {
      axes: new Attribute('axes', 'INTS', options?.axes || []),
      keepdims: new Attribute('keepdims', 'INT', options?.keepDimensions === false ? 0 : 1),
    });
  }
  reduceLogSumExp(input: MLOperand, options?: MLReduceOptions): MLOperand {
    return this.emitNode('ReduceLogSumExp', [input], {
      axes: new Attribute('axes', 'INTS', options?.axes || []),
      keepdims: new Attribute('keepdims', 'INT', options?.keepDimensions === false ? 0 : 1),
    });
  }
  argMax(input: MLOperand, options?: MLReduceOptions): MLOperand {
    return this.emitNode('ArgMax', [input], {
      axis: new Attribute('axis', 'INT', options?.axes?.[0] || 0),
      keepdims: new Attribute('keepdims', 'INT', options?.keepDimensions === false ? 0 : 1),
    });
  }
  argMin(input: MLOperand, options?: MLReduceOptions): MLOperand {
    return this.emitNode('ArgMin', [input], {
      axis: new Attribute('axis', 'INT', options?.axes?.[0] || 0),
      keepdims: new Attribute('keepdims', 'INT', options?.keepDimensions === false ? 0 : 1),
    });
  }

  // Phase 7: Normalization Operations
  batchNormalization(
    input: MLOperand,
    mean: MLOperand,
    variance: MLOperand,
    options?: MLBatchNormalizationOptions,
  ): MLOperand {
    const scale =
      options?.scale ||
      this.constant(
        { dataType: input.dataType, dimensions: mean.shape },
        new Float32Array(mean.shape.reduce((a, b) => a * b, 1)).fill(1),
      );
    const bias =
      options?.bias ||
      this.constant(
        { dataType: input.dataType, dimensions: mean.shape },
        new Float32Array(mean.shape.reduce((a, b) => a * b, 1)).fill(0),
      );
    return this.emitNode('BatchNormalization', [input, scale, bias, mean, variance], {
      epsilon: new Attribute('epsilon', 'FLOAT', options?.epsilon || 1e-5),
    });
  }
  instanceNormalization(input: MLOperand, options?: MLInstanceNormalizationOptions): MLOperand {
    const cDim = input.shape[1] || 1;
    const scale =
      options?.scale ||
      this.constant(
        { dataType: input.dataType, dimensions: [cDim] },
        new Float32Array(cDim).fill(1),
      );
    const bias =
      options?.bias ||
      this.constant(
        { dataType: input.dataType, dimensions: [cDim] },
        new Float32Array(cDim).fill(0),
      );
    return this.emitNode('InstanceNormalization', [input, scale, bias], {
      epsilon: new Attribute('epsilon', 'FLOAT', options?.epsilon || 1e-5),
    });
  }
  layerNormalization(input: MLOperand, options?: MLLayerNormalizationOptions): MLOperand {
    const cDim = input.shape[input.shape.length - 1] || 1;
    const scale =
      options?.scale ||
      this.constant(
        { dataType: input.dataType, dimensions: [cDim] },
        new Float32Array(cDim).fill(1),
      );
    const bias =
      options?.bias ||
      this.constant(
        { dataType: input.dataType, dimensions: [cDim] },
        new Float32Array(cDim).fill(0),
      );
    return this.emitNode('LayerNormalization', [input, scale, bias], {
      axis: new Attribute('axis', 'INT', options?.axes?.[0] || -1),
      epsilon: new Attribute('epsilon', 'FLOAT', options?.epsilon || 1e-5),
    });
  }
  l2Normalization(input: MLOperand, options?: MLReduceOptions): MLOperand {
    return this.emitNode('LpNormalization', [input], {
      axis: new Attribute('axis', 'INT', options?.axes?.[0] || -1),
      p: new Attribute('p', 'INT', 2),
    });
  }

  // Phase 11: Logical & Relational Operations
  equal(a: MLOperand, b: MLOperand): MLOperand {
    return this.emitNode('Equal', [a, b]);
  }
  greater(a: MLOperand, b: MLOperand): MLOperand {
    return this.emitNode('Greater', [a, b]);
  }
  greaterOrEqual(a: MLOperand, b: MLOperand): MLOperand {
    return this.emitNode('GreaterOrEqual', [a, b]);
  }
  lesser(a: MLOperand, b: MLOperand): MLOperand {
    return this.emitNode('Less', [a, b]);
  }
  lesserOrEqual(a: MLOperand, b: MLOperand): MLOperand {
    return this.emitNode('LessOrEqual', [a, b]);
  }
  logicalNot(a: MLOperand): MLOperand {
    return this.emitNode('Not', [a]);
  }
  logicalAnd(a: MLOperand, b: MLOperand): MLOperand {
    return this.emitNode('And', [a, b]);
  }
  logicalOr(a: MLOperand, b: MLOperand): MLOperand {
    return this.emitNode('Or', [a, b]);
  }
  logicalXor(a: MLOperand, b: MLOperand): MLOperand {
    return this.emitNode('Xor', [a, b]);
  }
  where(condition: MLOperand, trueValue: MLOperand, falseValue: MLOperand): MLOperand {
    return this.emitNode('Where', [condition, trueValue, falseValue]);
  }

  // Phase 16: Transformers & Quantization
  lstmCell(
    input: MLOperand,
    weight: MLOperand,
    recurrentWeight: MLOperand,
    hiddenState: MLOperand,
    cellState: MLOperand,
    hiddenSize: number,
    options?: any,
  ): MLOperand[] {
    const node = this.emitNode('LSTM', [
      input,
      weight,
      recurrentWeight,
      options?.bias,
      options?.sequenceLens,
      options?.initialH,
      options?.initialC,
      options?.P,
    ]);
    return [node, node, node]; // Return array for polyfill
  }
  gruCell(
    input: MLOperand,
    weight: MLOperand,
    recurrentWeight: MLOperand,
    hiddenState: MLOperand,
    hiddenSize: number,
    options?: any,
  ): MLOperand[] {
    const node = this.emitNode('GRU', [
      input,
      weight,
      recurrentWeight,
      options?.bias,
      options?.sequenceLens,
      options?.initialH,
    ]);
    return [node, node];
  }

  triangular(input: MLOperand, options?: { upper?: boolean; diagonal?: number }): MLOperand {
    // Emulated via Trilu
    return this.emitNode('Trilu', [input], {
      upper: new Attribute('upper', 'INT', options?.upper ? 1 : 0),
    });
  }
  scaledDotProductAttention(
    query: MLOperand,
    key: MLOperand,
    value: MLOperand,
    options?: any,
  ): MLOperand {
    return this.emitNode('Attention', [query, key, value]);
  }
  quantizeLinear(input: MLOperand, scale: MLOperand, zeroPoint: MLOperand): MLOperand {
    return this.emitNode('QuantizeLinear', [input, scale, zeroPoint]);
  }
  dequantizeLinear(input: MLOperand, scale: MLOperand, zeroPoint: MLOperand): MLOperand {
    return this.emitNode('DequantizeLinear', [input, scale, zeroPoint]);
  }
  bitwiseAnd(a: MLOperand, b: MLOperand): MLOperand {
    return this.emitNode('BitwiseAnd', [a, b]);
  }
  shiftRightLogical(a: MLOperand, b: MLOperand): MLOperand {
    return this.emitNode('BitShift', [a, b], {
      direction: new Attribute('direction', 'STRING', 'RIGHT'),
    });
  }
}
