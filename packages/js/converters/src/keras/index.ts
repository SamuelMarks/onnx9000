import { parseTFJSModel, JsonObject } from './tfjs-parser.js';
import { extractKerasTopology, KerasModelTopology, KerasNodeSpec } from './keras-ast.js';
import { emitConv } from './emitters-conv.js';
import { emitDense, emitActivation, emitIdentity, OnnxNodeBuilder } from './emitters.js';
import { emitPool, emitGlobalPool } from './emitters-pool.js';
import { LayoutOptimizer, OnnxNodeLike } from './layout-optimizer.js';
import { KerasGraphOptimizer } from './optimizers.js';
import { getCustomKerasLayerEmitter } from './plugin-registry.js';
import {
  Graph,
  Node,
  ValueInfo,
  serializeModelProto,
  Attribute,
  Tensor,
  Shape,
  AttributeType,
  AttributeValue,
} from '@onnx9000/core';

/**
 * Type definition for a node translation handler.
 */
type Handler = (
  inName: string,
  outName: string,
  layerName: string,
  nodeName: string,
  config: JsonObject,
  className: string,
  node: KerasNodeSpec,
) => OnnxNodeBuilder[];

/**
 * The Keras2OnnxConverter translates Keras (Layers or Sequential) models into ONNX Core IR.
 */
export class Keras2OnnxConverter {
  /**
   * The extracted Keras model topology.
   */
  private topology: KerasModelTopology;

  /**
   * Intermediate representation of nodes before they are mapped to core ONNX nodes.
   */
  private rawNodes: OnnxNodeBuilder[] = [];

  /**
   * Optimizer for handling memory layout (NHWC vs NCHW).
   */
  private layoutOptimizer = new LayoutOptimizer();

  /**
   * Registry of handlers for different Keras layer types.
   */
  private handlers = new Map<string, Handler>();

  /**
   * The final list of nodes generated after all passes.
   */
  private _finalNodes: Node[] = [];

  /**
   * Initializes a new instance of the Keras2OnnxConverter class.
   * @param modelJson The Keras model topology in JSON format.
   */
  constructor(modelJson: string) {
    const model = parseTFJSModel(modelJson);
    const signature = model.format === 'layers-model' ? model.signature : undefined;
    this.topology = extractKerasTopology(model.modelTopology, '', signature || {});
    this.registerHandlers();
  }

  /**
   * Gets the final list of nodes used for testing.
   * @returns The list of ONNX nodes.
   */
  public get _test_finalNodes(): Node[] {
    return this._finalNodes;
  }

  /**
   * Registers all layer translation handlers into the registry.
   */
  private registerHandlers(): void {
    this.handlers.set('Dense', this.handleDense.bind(this));
    this.handlers.set('QDense', this.handleDense.bind(this));
    this.handlers.set('Activation', this.handleActivation.bind(this));
    this.handlers.set('QActivation', this.handleActivation.bind(this));
    this.handlers.set('LeakyReLU', this.handleLeakyReLU.bind(this));
    this.handlers.set('PReLU', this.handlePReLU.bind(this));
    this.handlers.set('ELU', this.handleELU.bind(this));
    this.handlers.set('ThresholdedReLU', this.handleThresholdedReLU.bind(this));
    this.handlers.set('Softmax', this.handleSoftmax.bind(this));
    this.handlers.set('Conv1D', this.handleConv2D.bind(this));
    this.handlers.set('Conv2D', this.handleConv2D.bind(this));
    this.handlers.set('Conv3D', this.handleConv2D.bind(this));
    this.handlers.set('QConv1D', this.handleConv2D.bind(this));
    this.handlers.set('QConv2D', this.handleConv2D.bind(this));
    this.handlers.set('QConv3D', this.handleConv2D.bind(this));
    this.handlers.set('MaxPooling1D', this.handlePooling2D.bind(this));
    this.handlers.set('MaxPooling2D', this.handlePooling2D.bind(this));
    this.handlers.set('MaxPooling3D', this.handlePooling2D.bind(this));
    this.handlers.set('AveragePooling1D', this.handlePooling2D.bind(this));
    this.handlers.set('AveragePooling2D', this.handlePooling2D.bind(this));
    this.handlers.set('AveragePooling3D', this.handlePooling2D.bind(this));
    this.handlers.set('GlobalAveragePooling1D', this.handleGlobalPooling2D.bind(this));
    this.handlers.set('GlobalAveragePooling2D', this.handleGlobalPooling2D.bind(this));
    this.handlers.set('GlobalAveragePooling3D', this.handleGlobalPooling2D.bind(this));
    this.handlers.set('GlobalMaxPooling1D', this.handleGlobalPooling2D.bind(this));
    this.handlers.set('GlobalMaxPooling2D', this.handleGlobalPooling2D.bind(this));
    this.handlers.set('GlobalMaxPooling3D', this.handleGlobalPooling2D.bind(this));
    this.handlers.set('BatchNormalization', this.handleBatchNormalization.bind(this));
    this.handlers.set('LayerNormalization', this.handleLayerNormalization.bind(this));
    this.handlers.set('UnitNormalization', this.handleUnitNormalization.bind(this));
    this.handlers.set('GroupNormalization', this.handleGroupNormalization.bind(this));
    this.handlers.set('Embedding', this.handleEmbedding.bind(this));

    const identityLayers = [
      'GaussianNoise',
      'GaussianDropout',
      'AlphaDropout',
      'SpatialDropout1D',
      'SpatialDropout2D',
      'SpatialDropout3D',
      'ActivityRegularization',
      'Dropout',
      'RandomFlip',
      'RandomRotation',
      'RandomZoom',
      'RandomCrop',
      'RandomTranslation',
      'RandomContrast',
      'RandomBrightness',
    ];
    for (const layer of identityLayers) {
      this.handlers.set(layer, (inName, outName, _layerName, nodeName) =>
        emitIdentity(inName, outName, nodeName),
      );
    }

    this.handlers.set('Permute', this.handlePermute.bind(this));
    this.handlers.set('Flatten', this.handleFlatten.bind(this));
    this.handlers.set('Reshape', this.handleReshape.bind(this));
    this.handlers.set('Rescaling', this.handleRescaling.bind(this));
    this.handlers.set('Resizing', this.handleResizing.bind(this));
    this.handlers.set('CenterCrop', this.handleCenterCrop.bind(this));

    const mathLayers = ['Add', 'Subtract', 'Multiply', 'Minimum', 'Maximum'];
    for (const layer of mathLayers) {
      this.handlers.set(layer, this.handleMath.bind(this));
    }

    this.handlers.set('Concatenate', this.handleConcatenate.bind(this));
    this.handlers.set('Average', this.handleAverage.bind(this));
    this.handlers.set('Dot', this.handleDot.bind(this));
    this.handlers.set('EinsumDense', this.handleEinsumDense.bind(this));
  }

  /**
   * Translates a Keras node into one or more intermediate ONNX nodes.
   * @param node The Keras node specification.
   */
  private translateNode(node: KerasNodeSpec): void {
    const className = node.className;
    const config = node.config;
    const nodeName = node.name;
    const layerName = node.layerName;

    const inName =
      node.inboundNodes.length > 0 ? (node.inboundNodes[0] as string) : `${nodeName}_input`;
    const outName = `${nodeName}:0`;

    const customEmitter = getCustomKerasLayerEmitter(className);
    if (customEmitter !== undefined) {
      this.rawNodes.push(...customEmitter(nodeName, layerName, node.inboundNodes, outName, config));
      return;
    }

    if (className === 'InputLayer') return;

    const handler = this.handlers.get(className);
    if (handler) {
      this.rawNodes.push(...handler(inName, outName, layerName, nodeName, config, className, node));
    }
  }

  /**
   * Converts the Keras model to an ONNX model, executing all required architectural passes.
   * @returns The serialized ONNX ModelProto as a Uint8Array.
   */
  public convert(): Uint8Array {
    const entries = Array.from(this.topology.nodes.values());
    for (const node of entries) {
      this.translateNode(node);
    }

    // Phase 10 Memory Layout & Dimension Resolution Pass
    const spatialOps = [
      'Conv',
      'QLinearConv',
      'MaxPool',
      'AveragePool',
      'GlobalMaxPool',
      'GlobalAveragePool',
      'Resize',
      'BatchNormalization',
      'InstanceNormalization',
      'GroupNormalization',
    ];
    const layoutNodes: OnnxNodeBuilder[] = [];

    for (const rn of this.rawNodes) {
      let requiresLayoutConversion = false;
      let rank = 4;

      for (const nSpec of this.topology.nodes.values()) {
        if (rn.name.startsWith(nSpec.name) || rn.name === nSpec.name) {
          if (nSpec.config.data_format === 'channels_last') {
            requiresLayoutConversion = true;
            if (nSpec.className.includes('1D')) rank = 3;
            else if (nSpec.className.includes('2D')) rank = 4;
            else if (nSpec.className.includes('3D')) rank = 5;
          }
          break;
        }
      }

      if (requiresLayoutConversion && spatialOps.includes(rn.opType)) {
        let toNchwPerm: number[];
        let toNhwcPerm: number[];

        if (rank === 3) {
          toNchwPerm = [0, 2, 1];
          toNhwcPerm = [0, 2, 1];
        } else if (rank === 5) {
          toNchwPerm = [0, 4, 1, 2, 3];
          toNhwcPerm = [0, 2, 3, 4, 1];
        } else {
          toNchwPerm = [0, 3, 1, 2];
          toNhwcPerm = [0, 2, 3, 1];
        }

        const originalInput = rn.inputs[0]!;
        const nchwInputName = `${originalInput}_to_nchw`;

        layoutNodes.push({
          opType: 'Transpose',
          inputs: [originalInput],
          outputs: [nchwInputName],
          name: `${rn.name}_nchw_in`,
          attributes: [{ name: 'perm', ints: toNchwPerm, type: 'INTS' }],
        });

        rn.inputs[0] = nchwInputName;

        if (rn.opType === 'Conv' || rn.opType === 'QLinearConv') {
          const weightIdx = rn.opType === 'QLinearConv' ? 3 : 1;
          if (rn.inputs.length > weightIdx && rn.inputs[weightIdx]) {
            const originalWeight = rn.inputs[weightIdx];
            const hwioToOihwPerm =
              rank === 3 ? [2, 1, 0] : rank === 5 ? [4, 3, 0, 1, 2] : [3, 2, 0, 1];
            const transposedWeightName = `${originalWeight}_oihw`;

            layoutNodes.push({
              opType: 'Transpose',
              inputs: [originalWeight],
              outputs: [transposedWeightName],
              name: `${rn.name}_weight_oihw`,
              attributes: [{ name: 'perm', ints: hwioToOihwPerm, type: 'INTS' }],
            });

            rn.inputs[weightIdx] = transposedWeightName;
          }
        }

        const originalOutput = rn.outputs[0]!;
        const nchwOutputName = `${originalOutput}_nchw`;

        rn.outputs[0] = nchwOutputName;
        layoutNodes.push(rn);

        layoutNodes.push({
          opType: 'Transpose',
          inputs: [nchwOutputName],
          outputs: [originalOutput],
          name: `${rn.name}_nhwc_out`,
          attributes: [{ name: 'perm', ints: toNhwcPerm, type: 'INTS' }],
        });
      } else {
        layoutNodes.push(rn);
      }
    }
    this.rawNodes = layoutNodes;

    // Type Preservation Pass (Phase 9 Mixed Precision)
    const castedNodes: OnnxNodeBuilder[] = [];
    for (const rn of this.rawNodes) {
      let requiresF16 = false;
      for (const nSpec of this.topology.nodes.values()) {
        if (
          rn.name.startsWith(nSpec.name) &&
          (nSpec.config.dtype === 'mixed_float16' || nSpec.config.dtype === 'float16')
        ) {
          requiresF16 = true;
          break;
        }
      }

      if (requiresF16) {
        const castedInputs = rn.inputs.map((inp: string) => {
          if (inp.includes('_weight') || inp.includes('_bias') || inp.includes('_kernel'))
            return inp;
          const castName = `${inp}_cast_to_f16`;
          castedNodes.push({
            opType: 'Cast',
            inputs: [inp],
            outputs: [castName],
            name: `${rn.name}_cast_in`,
            attributes: [{ name: 'to', i: 10, type: 'INT' }],
          });
          return castName;
        });
        rn.inputs = castedInputs;
      }
      castedNodes.push(rn);
    }
    this.rawNodes = castedNodes;

    // Optimize layout
    this.rawNodes = this.layoutOptimizer.optimize(
      this.rawNodes as OnnxNodeLike[],
    ) as OnnxNodeBuilder[];

    // Map OnnxNodeBuilder to @onnx9000/core Node
    const coreNodes: Node[] = this.rawNodes.map((rn) => {
      const attributes: Record<string, Attribute> = {};
      for (const attr of rn.attributes) {
        let type: AttributeType = 'UNKNOWN';
        let val: AttributeValue = null;
        if (attr.type === 'INT' || attr.i !== undefined) {
          type = 'INT';
          val = attr.i ?? 0;
        } else if (attr.type === 'FLOAT' || attr.f !== undefined) {
          type = 'FLOAT';
          val = attr.f ?? 0.0;
        } else if (attr.type === 'STRING' || attr.s !== undefined) {
          type = 'STRING';
          val = attr.s ?? '';
        } else if (attr.type === 'INTS' || attr.ints !== undefined) {
          type = 'INTS';
          val = attr.ints ?? [];
        } else if (attr.type === 'FLOATS' || attr.floats !== undefined) {
          type = 'FLOATS';
          val = attr.floats ?? [];
        }
        attributes[attr.name] = new Attribute(attr.name, type, val);
      }
      return new Node(rn.opType, rn.inputs, rn.outputs, attributes, rn.name);
    });

    // QAT (Quantization-Aware Training) Pass
    const qatNodes: Node[] = [];
    for (const node of coreNodes) {
      if (node.name.includes('quantize_wrapper') || node.name.includes('qat_')) {
        const scaleName = `${node.name}_qat_scale`;
        const zpName = `${node.name}_qat_zp`;
        const input0 = node.inputs[0] || '';
        const qOutName = `${input0}_quantized`;
        const dqOutName = `${input0}_dequantized`;

        qatNodes.push(
          new Node('QuantizeLinear', [input0, scaleName, zpName], [qOutName], {}, `${node.name}_q`),
        );
        qatNodes.push(
          new Node(
            'DequantizeLinear',
            [qOutName, scaleName, zpName],
            [dqOutName],
            {},
            `${node.name}_dq`,
          ),
        );

        node.inputs[0] = dqOutName;
      }
      qatNodes.push(node);
    }

    // DynamicQuantizeLinear (AWQ / GPTQ) Pass
    const finalNodesAfterQuant: Node[] = [];
    for (const node of qatNodes) {
      if (node.opType === 'MatMul' && node.name.includes('packed_4bit')) {
        node.opType = 'MatMulNBits';
      } else if (node.opType === 'MatMul' && node.name.includes('dynamic_quant')) {
        const input0 = node.inputs[0] || '';
        const dQuantOut = `${input0}_dyn_quant`;
        const scaleOut = `${input0}_dyn_scale`;
        const zpOut = `${input0}_dyn_zp`;
        finalNodesAfterQuant.push(
          new Node(
            'DynamicQuantizeLinear',
            [input0],
            [dQuantOut, scaleOut, zpOut],
            {},
            `${node.name}_dyn_q`,
          ),
        );

        node.opType = 'MatMulInteger';
        const input1 = node.inputs[1] || '';
        node.inputs = [dQuantOut, input1, zpOut, input1 ? `${input1}_zp` : ''];
      }
      finalNodesAfterQuant.push(node);
    }

    // Explicit Masking Propagation (Phase 10)
    const maskedNodes: Node[] = [];
    for (const node of finalNodesAfterQuant) {
      maskedNodes.push(node);
      if (node.opType === 'Gather' && node.name.includes('embed_masking')) {
        const inputTensor = node.inputs[1] as string;
        const maskOutName = `${node.name}_keras_mask`;
        const zeroTensorName = `${node.name}_zero_const`;

        maskedNodes.push(
          new Node(
            'Constant',
            [],
            [zeroTensorName],
            { value: new Attribute('value', 'INT', 0) },
            `${node.name}_mask_zero`,
          ),
        );
        maskedNodes.push(
          new Node(
            'Equal',
            [inputTensor, zeroTensorName],
            [`${node.name}_is_zero`],
            {},
            `${node.name}_mask_eq`,
          ),
        );
        maskedNodes.push(
          new Node('Not', [`${node.name}_is_zero`], [maskOutName], {}, `${node.name}_mask_not`),
        );
      }
    }

    const graph = new Graph('keras_to_onnx_model');
    graph.nodes = maskedNodes;

    const optimizer = new KerasGraphOptimizer();
    optimizer.optimize(graph);

    const allOutputs = new Set<string>();
    for (const n of graph.nodes) {
      for (const out of n.outputs) {
        allOutputs.add(out);
      }
    }

    const inputs = new Set<string>();
    const outputs = new Set<string>();
    for (const n of graph.nodes) {
      for (const inp of n.inputs) {
        if (!allOutputs.has(inp)) {
          if (inp.includes('_weights') || inp.includes('_kernel') || inp.includes('_bias')) {
            if (!graph.initializers.includes(inp)) {
              graph.initializers.push(inp);
              const shape: Shape = inp.includes('_bias') ? [1] : [1, 1, 3, 3];
              const t = new Tensor(inp, shape, 'float32', true);
              t.data = new Uint8Array(4 * (shape.length === 1 ? 1 : 9));
              graph.tensors[inp] = t;
            }
          } else if (inp !== '') {
            inputs.add(inp);
          }
        }
      }
    }

    const allInputs = new Set<string>();
    for (const n of graph.nodes) {
      for (const inp of n.inputs) {
        allInputs.add(inp);
      }
    }
    for (const out of allOutputs) {
      if (!allInputs.has(out)) {
        outputs.add(out);
      }
    }

    for (const inp of inputs) {
      const topIn = this.topology.inputs.find((x) => x.name === inp);
      let shape: Shape = [-1, -1, -1, -1];
      if (topIn && topIn.shape && topIn.shape.length > 0) {
        shape = topIn.shape.map((s, idx) => {
          if (s === null) return idx === 0 ? 'batch_size' : -1;
          return s;
        });
      }

      let signatureName = inp;
      if (this.topology.signatures && this.topology.signatures['serving_default']) {
        for (const [sName, internalName] of Object.entries(
          this.topology.signatures['serving_default'].inputs,
        )) {
          if (internalName === inp || internalName === inp.split(':')[0]) {
            signatureName = sName;
          }
        }
      }
      graph.inputs.push(new ValueInfo(signatureName, shape, 'float32'));

      if (signatureName !== inp) {
        for (const n of graph.nodes) {
          for (let j = 0; j < n.inputs.length; j++) {
            if (n.inputs[j] === inp) n.inputs[j] = signatureName;
          }
        }
      }
    }

    for (const out of outputs) {
      const topOut = this.topology.outputs.find((x) => x.name === out);
      let shape: Shape = [-1, -1];
      if (topOut && topOut.shape && topOut.shape.length > 0) {
        shape = topOut.shape.map((s, idx) => {
          if (s === null) return idx === 0 ? 'batch_size' : -1;
          return s;
        });
      }

      let signatureName = out;
      if (this.topology.signatures && this.topology.signatures['serving_default']) {
        for (const [sName, internalName] of Object.entries(
          this.topology.signatures['serving_default'].outputs,
        )) {
          if (internalName === out || internalName === out.split(':')[0]) {
            signatureName = sName;
          }
        }
      }
      graph.outputs.push(new ValueInfo(signatureName, shape, 'float32'));

      if (signatureName !== out) {
        for (const n of graph.nodes) {
          for (let j = 0; j < n.outputs.length; j++) {
            if (n.outputs[j] === out) n.outputs[j] = signatureName;
          }
        }
      }
    }

    this._finalNodes = graph.nodes;
    return serializeModelProto(graph);
  }

  /**
   * Handles translation of Dense and QDense layers.
   */
  private handleDense(
    inName: string,
    outName: string,
    layerName: string,
    nodeName: string,
    config: JsonObject,
    className: string,
  ): OnnxNodeBuilder[] {
    const activation = (config.activation as string) || 'linear';
    const useBias = config.use_bias !== false;

    if (className === 'QDense') {
      return [
        {
          opType: 'QLinearMatMul',
          inputs: [
            inName,
            `${inName}_scale`,
            `${inName}_zp`,
            `${layerName}_weights`,
            `${layerName}_weights_scale`,
            `${layerName}_weights_zp`,
            `${outName}_scale`,
            `${outName}_zp`,
          ],
          outputs: [outName],
          name: nodeName,
          attributes: [],
        },
      ];
    }
    return emitDense(
      inName,
      outName,
      `${layerName}_weights`,
      useBias ? `${layerName}_bias` : undefined,
      activation,
      nodeName,
    );
  }

  /**
   * Handles translation of Activation and QActivation layers.
   */
  private handleActivation(
    inName: string,
    outName: string,
    _layerName: string,
    nodeName: string,
    config: JsonObject,
  ): OnnxNodeBuilder[] {
    const activation = config.activation as string;
    return emitActivation(activation, inName, outName, nodeName);
  }

  /**
   * Handles translation of LeakyReLU layers.
   */
  private handleLeakyReLU(
    inName: string,
    outName: string,
    _layerName: string,
    nodeName: string,
    config: JsonObject,
  ): OnnxNodeBuilder[] {
    const alpha = typeof config.alpha === 'number' ? config.alpha : 0.3;
    return emitActivation('leaky_relu', inName, outName, nodeName, { alpha });
  }

  /**
   * Handles translation of PReLU layers.
   */
  private handlePReLU(
    inName: string,
    outName: string,
    layerName: string,
    nodeName: string,
  ): OnnxNodeBuilder[] {
    return emitActivation('prelu', inName, outName, nodeName, {
      alphaWeightName: `${layerName}_alpha`,
    });
  }

  /**
   * Handles translation of ELU layers.
   */
  private handleELU(
    inName: string,
    outName: string,
    _layerName: string,
    nodeName: string,
    config: JsonObject,
  ): OnnxNodeBuilder[] {
    const alpha = typeof config.alpha === 'number' ? config.alpha : 1.0;
    return emitActivation('elu', inName, outName, nodeName, { alpha });
  }

  /**
   * Handles translation of ThresholdedReLU layers.
   */
  private handleThresholdedReLU(
    inName: string,
    outName: string,
    _layerName: string,
    nodeName: string,
    config: JsonObject,
  ): OnnxNodeBuilder[] {
    const theta = typeof config.theta === 'number' ? config.theta : 1.0;
    return emitActivation('thresholded_relu', inName, outName, nodeName, { theta });
  }

  /**
   * Handles translation of Softmax layers.
   */
  private handleSoftmax(
    inName: string,
    outName: string,
    _layerName: string,
    nodeName: string,
    config: JsonObject,
  ): OnnxNodeBuilder[] {
    const axis = typeof config.axis === 'number' ? config.axis : -1;
    return [
      {
        opType: 'Softmax',
        inputs: [inName],
        outputs: [outName],
        name: nodeName,
        attributes: [{ name: 'axis', i: axis, type: 'INT' }],
      },
    ];
  }

  /**
   * Handles translation of Conv2D and QConv2D layers.
   */
  private handleConv2D(
    inName: string,
    outName: string,
    layerName: string,
    nodeName: string,
    config: JsonObject,
    className: string,
  ): OnnxNodeBuilder[] {
    const activation = (config.activation as string) || 'linear';
    const padding = (config.padding as string) === 'same' ? 'same' : 'valid';
    const strides = (config.strides as number[] | undefined) ?? [1, 1];
    const dilation_rate = (config.dilation_rate as number[] | undefined) ?? [1, 1];
    const kernel_size = (config.kernel_size as number[] | undefined) ?? [1, 1];

    if (className === 'QConv2D') {
      return [
        {
          opType: 'QLinearConv',
          inputs: [
            inName,
            `${inName}_scale`,
            `${inName}_zp`,
            `${layerName}_kernel`,
            `${layerName}_kernel_scale`,
            `${layerName}_kernel_zp`,
            `${outName}_scale`,
            `${outName}_zp`,
          ],
          outputs: [outName],
          name: nodeName,
          attributes: [
            { name: 'strides', ints: strides, type: 'INTS' },
            { name: 'dilations', ints: dilation_rate, type: 'INTS' },
            { name: 'kernel_shape', ints: kernel_size, type: 'INTS' },
          ],
        },
      ];
    }
    return emitConv('Conv', inName, outName, `${layerName}_kernel`, `${layerName}_bias`, nodeName, {
      activation,
      padding,
      strides,
      dilations: dilation_rate,
      kernelShape: kernel_size,
    });
  }

  /**
   * Handles translation of Pooling2D layers.
   */
  private handlePooling2D(
    inName: string,
    outName: string,
    _layerName: string,
    nodeName: string,
    config: JsonObject,
    className: string,
  ): OnnxNodeBuilder[] {
    const isMax = className.startsWith('Max');
    const padding = (config.padding as string) === 'same' ? 'same' : 'valid';
    const pool_size = (config.pool_size as number[] | undefined) ?? [2, 2];
    const strides = (config.strides as number[] | undefined) ?? [2, 2];

    return emitPool(isMax ? 'Max' : 'Average', inName, outName, nodeName, {
      padding,
      poolSize: pool_size,
      strides,
    });
  }

  /**
   * Handles translation of GlobalPooling2D layers.
   */
  private handleGlobalPooling2D(
    inName: string,
    outName: string,
    _layerName: string,
    nodeName: string,
    config: JsonObject,
    className: string,
  ): OnnxNodeBuilder[] {
    const isMax = className.startsWith('GlobalMax');
    const keepDims = config.keepdims === true;
    return emitGlobalPool(isMax ? 'Max' : 'Average', inName, outName, nodeName, { keepDims });
  }

  /**
   * Handles translation of BatchNormalization layers.
   */
  private handleBatchNormalization(
    inName: string,
    outName: string,
    layerName: string,
    nodeName: string,
    config: JsonObject,
  ): OnnxNodeBuilder[] {
    const epsilon = typeof config.epsilon === 'number' ? config.epsilon : 1e-3;
    const momentum = typeof config.momentum === 'number' ? config.momentum : 0.99;
    const scale = config.scale !== false;
    const center = config.center !== false;

    const gammaName = scale ? `${layerName}_gamma` : '';
    const betaName = center ? `${layerName}_beta` : '';
    const meanName = `${layerName}_moving_mean`;
    const varName = `${layerName}_moving_variance`;

    return [
      {
        opType: 'BatchNormalization',
        inputs: [inName, gammaName, betaName, meanName, varName],
        outputs: [outName],
        name: nodeName,
        attributes: [
          { name: 'epsilon', f: epsilon, type: 'FLOAT' },
          { name: 'momentum', f: momentum, type: 'FLOAT' },
        ],
      },
    ];
  }

  /**
   * Handles translation of LayerNormalization layers.
   */
  private handleLayerNormalization(
    inName: string,
    outName: string,
    layerName: string,
    nodeName: string,
    config: JsonObject,
  ): OnnxNodeBuilder[] {
    const epsilon = typeof config.epsilon === 'number' ? config.epsilon : 1e-3;
    const axis = typeof config.axis === 'number' ? config.axis : -1;
    const scale = config.scale !== false;
    const center = config.center !== false;

    const inputs = [inName];
    if (scale) inputs.push(`${layerName}_gamma`);
    if (center) inputs.push(`${layerName}_beta`);

    return [
      {
        opType: 'LayerNormalization',
        inputs,
        outputs: [outName],
        name: nodeName,
        attributes: [
          { name: 'axis', i: axis, type: 'INT' },
          { name: 'epsilon', f: epsilon, type: 'FLOAT' },
        ],
      },
    ];
  }

  /**
   * Handles translation of UnitNormalization layers.
   */
  private handleUnitNormalization(
    inName: string,
    outName: string,
    _layerName: string,
    nodeName: string,
    config: JsonObject,
  ): OnnxNodeBuilder[] {
    const axis = typeof config.axis === 'number' ? config.axis : -1;
    return [
      {
        opType: 'LpNormalization',
        inputs: [inName],
        outputs: [outName],
        name: nodeName,
        attributes: [
          { name: 'axis', i: axis, type: 'INT' },
          { name: 'p', i: 2, type: 'INT' },
        ],
      },
    ];
  }

  /**
   * Handles translation of GroupNormalization layers.
   */
  private handleGroupNormalization(
    inName: string,
    outName: string,
    layerName: string,
    nodeName: string,
    config: JsonObject,
  ): OnnxNodeBuilder[] {
    const epsilon = typeof config.epsilon === 'number' ? config.epsilon : 1e-3;
    const groups = typeof config.groups === 'number' ? config.groups : 32;
    const scale = config.scale !== false;
    const center = config.center !== false;

    const inputs = [inName];
    inputs.push(scale ? `${layerName}_gamma` : '');
    inputs.push(center ? `${layerName}_beta` : '');

    return [
      {
        opType: 'GroupNormalization',
        inputs,
        outputs: [outName],
        name: nodeName,
        attributes: [
          { name: 'epsilon', f: epsilon, type: 'FLOAT' },
          { name: 'num_groups', i: groups, type: 'INT' },
        ],
      },
    ];
  }

  /**
   * Handles translation of Embedding layers.
   */
  private handleEmbedding(
    inName: string,
    outName: string,
    layerName: string,
    nodeName: string,
  ): OnnxNodeBuilder[] {
    return [
      {
        opType: 'Gather',
        inputs: [`${layerName}_weights`, inName],
        outputs: [outName],
        name: nodeName,
        attributes: [{ name: 'axis', i: 0, type: 'INT' }],
      },
    ];
  }

  /**
   * Handles translation of Permute layers.
   */
  private handlePermute(
    inName: string,
    outName: string,
    _layerName: string,
    nodeName: string,
    config: JsonObject,
  ): OnnxNodeBuilder[] {
    const dims = (config.dims as number[] | undefined) ?? [];
    const perm = [0, ...dims];
    return [
      {
        opType: 'Transpose',
        inputs: [inName],
        outputs: [outName],
        name: nodeName,
        attributes: [{ name: 'perm', ints: perm, type: 'INTS' }],
      },
    ];
  }

  /**
   * Handles translation of Flatten layers.
   */
  private handleFlatten(
    inName: string,
    outName: string,
    _layerName: string,
    nodeName: string,
  ): OnnxNodeBuilder[] {
    return [
      {
        opType: 'Flatten',
        inputs: [inName],
        outputs: [outName],
        name: nodeName,
        attributes: [{ name: 'axis', i: 1, type: 'INT' }],
      },
    ];
  }

  /**
   * Handles translation of Reshape layers.
   */
  private handleReshape(
    inName: string,
    outName: string,
    layerName: string,
    nodeName: string,
    config: JsonObject,
  ): OnnxNodeBuilder[] {
    const target_shape = (config.target_shape as (number | null)[] | undefined) ?? [];
    const targetShape = [0, ...target_shape.map((s) => (s === null ? -1 : s))];
    const shapeTensorName = `${layerName}_shape`;

    return [
      {
        opType: 'Constant',
        inputs: [],
        outputs: [shapeTensorName],
        name: `${nodeName}_shape_const`,
        attributes: [{ name: 'value', ints: targetShape, type: 'INTS' }],
      },
      {
        opType: 'Reshape',
        inputs: [inName, shapeTensorName],
        outputs: [outName],
        name: nodeName,
        attributes: [],
      },
    ];
  }

  /**
   * Handles translation of Rescaling layers.
   */
  private handleRescaling(
    inName: string,
    outName: string,
    layerName: string,
    nodeName: string,
    config: JsonObject,
  ): OnnxNodeBuilder[] {
    const scale = typeof config.scale === 'number' ? config.scale : 1.0;
    const offset = typeof config.offset === 'number' ? config.offset : 0.0;

    const scaleName = `${layerName}_scale`;
    const offsetName = `${layerName}_offset`;
    const mulOut = `${nodeName}_mul`;

    return [
      {
        opType: 'Constant',
        inputs: [],
        outputs: [scaleName],
        name: `${nodeName}_scale_const`,
        attributes: [{ name: 'value', f: scale, type: 'FLOAT' }],
      },
      {
        opType: 'Constant',
        inputs: [],
        outputs: [offsetName],
        name: `${nodeName}_offset_const`,
        attributes: [{ name: 'value', f: offset, type: 'FLOAT' }],
      },
      {
        opType: 'Mul',
        inputs: [inName, scaleName],
        outputs: [mulOut],
        name: `${nodeName}_mul`,
        attributes: [],
      },
      {
        opType: 'Add',
        inputs: [mulOut, offsetName],
        outputs: [outName],
        name: `${nodeName}_add`,
        attributes: [],
      },
    ];
  }

  /**
   * Handles translation of Resizing layers.
   */
  private handleResizing(
    inName: string,
    outName: string,
    layerName: string,
    nodeName: string,
    config: JsonObject,
  ): OnnxNodeBuilder[] {
    const height = (config.height as number) || 0;
    const width = (config.width as number) || 0;
    const interpolation = (config.interpolation as string) || 'bilinear';

    let mode = 'linear';
    if (interpolation === 'nearest') mode = 'nearest';
    else if (interpolation === 'bicubic') mode = 'cubic';

    const sizesName = `${layerName}_sizes`;
    const roiName = `${layerName}_roi`;
    const scalesName = `${layerName}_scales`;

    return [
      {
        opType: 'Constant',
        inputs: [],
        outputs: [sizesName],
        name: `${nodeName}_sizes_const`,
        attributes: [{ name: 'value', ints: [1, 1, height, width], type: 'INTS' }],
      },
      {
        opType: 'Constant',
        inputs: [],
        outputs: [roiName],
        name: `${nodeName}_roi_const`,
        attributes: [{ name: 'value', floats: [], type: 'FLOATS' }],
      },
      {
        opType: 'Constant',
        inputs: [],
        outputs: [scalesName],
        name: `${nodeName}_scales_const`,
        attributes: [{ name: 'value', floats: [], type: 'FLOATS' }],
      },
      {
        opType: 'Resize',
        inputs: [inName, roiName, scalesName, sizesName],
        outputs: [outName],
        name: nodeName,
        attributes: [{ name: 'mode', s: mode, type: 'STRING' }],
      },
    ];
  }

  /**
   * Handles translation of CenterCrop layers.
   */
  private handleCenterCrop(
    inName: string,
    outName: string,
    _layerName: string,
    nodeName: string,
  ): OnnxNodeBuilder[] {
    return [
      {
        opType: 'Slice',
        inputs: [inName],
        outputs: [outName],
        name: nodeName,
        attributes: [],
      },
    ];
  }

  /**
   * Handles translation of mathematical layers (Add, Subtract, Multiply, Minimum, Maximum).
   */
  private handleMath(
    _inName: string,
    outName: string,
    _layerName: string,
    nodeName: string,
    _config: JsonObject,
    className: string,
    node: KerasNodeSpec,
  ): OnnxNodeBuilder[] {
    const opMap: Record<string, string> = {
      Add: 'Add',
      Subtract: 'Sub',
      Multiply: 'Mul',
      Minimum: 'Min',
      Maximum: 'Max',
    };

    const onnxOp = opMap[className] || 'Add';
    const nodes: OnnxNodeBuilder[] = [];

    if (node.inboundNodes.length === 1) {
      return emitIdentity(node.inboundNodes[0] as string, outName, nodeName);
    } else if (node.inboundNodes.length === 2 || ['Sub', 'Mul'].includes(onnxOp)) {
      let currentOut = node.inboundNodes[0] as string;
      for (let j = 1; j < node.inboundNodes.length; j++) {
        const nextIn = node.inboundNodes[j] as string;
        const iterOut =
          j === node.inboundNodes.length - 1 ? outName : `${nodeName}_merge_${j.toString()}`;
        nodes.push({
          opType: onnxOp,
          inputs: [currentOut, nextIn],
          outputs: [iterOut],
          name: `${nodeName}_${j.toString()}`,
          attributes: [],
        });
        currentOut = iterOut;
      }
    } else {
      const nOpMap: Record<string, string> = {
        Add: 'Sum',
        Minimum: 'Min',
        Maximum: 'Max',
      };
      nodes.push({
        opType: nOpMap[className] || 'Sum',
        inputs: [...node.inboundNodes],
        outputs: [outName],
        name: nodeName,
        attributes: [],
      });
    }
    return nodes;
  }

  /**
   * Handles translation of Concatenate layers.
   */
  private handleConcatenate(
    _inName: string,
    outName: string,
    _layerName: string,
    nodeName: string,
    config: JsonObject,
    _className: string,
    node: KerasNodeSpec,
  ): OnnxNodeBuilder[] {
    const axis = typeof config.axis === 'number' ? config.axis : -1;
    return [
      {
        opType: 'Concat',
        inputs: [...node.inboundNodes],
        outputs: [outName],
        name: nodeName,
        attributes: [{ name: 'axis', i: axis, type: 'INT' }],
      },
    ];
  }

  /**
   * Handles translation of Average layers.
   */
  private handleAverage(
    _inName: string,
    outName: string,
    _layerName: string,
    nodeName: string,
    _config: JsonObject,
    _className: string,
    node: KerasNodeSpec,
  ): OnnxNodeBuilder[] {
    if (node.inboundNodes.length === 1) {
      return emitIdentity(node.inboundNodes[0] as string, outName, nodeName);
    }
    return [
      {
        opType: 'Mean',
        inputs: [...node.inboundNodes],
        outputs: [outName],
        name: nodeName,
        attributes: [],
      },
    ];
  }

  /**
   * Handles translation of Dot layers.
   */
  private handleDot(
    _inName: string,
    outName: string,
    _layerName: string,
    nodeName: string,
    _config: JsonObject,
    _className: string,
    node: KerasNodeSpec,
  ): OnnxNodeBuilder[] {
    return [
      {
        opType: 'MatMul',
        inputs: [node.inboundNodes[0] as string, node.inboundNodes[1] as string],
        outputs: [outName],
        name: nodeName,
        attributes: [],
      },
    ];
  }

  /**
   * Handles translation of EinsumDense layers.
   */
  private handleEinsumDense(
    inName: string,
    outName: string,
    layerName: string,
    nodeName: string,
    config: JsonObject,
  ): OnnxNodeBuilder[] {
    const equation = config.equation as string;
    const nodes: OnnxNodeBuilder[] = [];
    nodes.push({
      opType: 'Einsum',
      inputs: [inName, `${layerName}_kernel`],
      outputs: [nodeName + '_einsum'],
      name: nodeName,
      attributes: [{ name: 'equation', s: equation, type: 'STRING' }],
    });
    if (config.bias_axes) {
      nodes.push({
        opType: 'Add',
        inputs: [nodeName + '_einsum', `${layerName}_bias`],
        outputs: [outName],
        name: nodeName + '_biasadd',
        attributes: [],
      });
    } else {
      nodes.push(...emitIdentity(nodeName + '_einsum', outName, nodeName + '_id'));
    }
    return nodes;
  }
}
