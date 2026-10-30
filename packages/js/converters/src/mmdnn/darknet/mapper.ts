/**
 * @fileoverview Darknet layer mapper to ONNX IR nodes and initializers.
 */
import { Attribute, type Graph, Node, Tensor, ValueInfo } from '@onnx9000/core';
import type { DarknetSection } from '../types.js';

/**
 * Maps Darknet configuration sections (`DarknetSection`) and weights into an ONNX `Graph`.
 */
export class DarknetMapper {
  /** Property `graph` of `DarknetMapper`. */
  private graph: Graph;
  /** Property `weights` of `DarknetMapper`. */
  private weights: Float32Array;
  /** Property `weightOffset` of `DarknetMapper`. */
  private weightOffset = 0;
  /** Property `layerOutputs` of `DarknetMapper`. */
  private layerOutputs: string[] = [];
  /** Property `channelsOutput` of `DarknetMapper`. */
  private channelsOutput: number[] = [];

  /**
   * Initializes a new `DarknetMapper`.
   * @param graph Target ONNX graph.
   * @param weights Parsed flat `Float32Array` of Darknet weights.
   */
  constructor(graph: Graph, weights: Float32Array) {
    this.graph = graph;
    this.weights = weights;
  }

  /**
   * Populates the ONNX graph with inputs, nodes, initializers, and outputs from Darknet sections.
   * @param layers Ordered array of Darknet sections parsed from `.cfg`.
   */
  public map(layers: DarknetSection[]): void {
    const netLayer = layers.find((l) => l.type === 'net');

    let currentChannels = netLayer?.channels || 3;
    const height = netLayer?.height || 416;
    const width = netLayer?.width || 416;

    let currentInput = 'input_0';
    this.graph.inputs.push(
      new ValueInfo(currentInput, ['batch_size', currentChannels, height, width], 'float32'),
    );

    const executionLayers = layers.filter((l) => l.type !== 'net');

    for (const [i, layer] of executionLayers.entries()) {
      const nodes = this.mapLayer(layer, i, currentInput, currentChannels);

      if (nodes.length > 0) {
        for (const n of nodes) {
          this.graph.addNode(n);
        }
        currentInput = (nodes[nodes.length - 1] as Node).outputs[0] as string;
      }

      this.layerOutputs.push(currentInput);

      if (layer.type === 'convolutional') {
        currentChannels = layer.filters || 1;
      } else if (layer.type === 'connected') {
        currentChannels = layer.output || 1;
      } else if (layer.type === 'route') {
        const rawRoute = layer.layers ?? 0;
        const routeLayers = Array.isArray(rawRoute) ? rawRoute : [rawRoute];
        currentChannels = 0;
        for (const r of routeLayers) {
          const idx = r < 0 ? i + r : r;
          currentChannels += this.channelsOutput[idx] || 0;
        }
      }

      this.channelsOutput.push(currentChannels);
    }

    const yoloLayers = executionLayers.filter((l) => l.type === 'yolo' || l.type === 'region');
    if (yoloLayers.length > 0) {
      for (const [i, l] of executionLayers.entries()) {
        if (l.type === 'yolo' || l.type === 'region') {
          const prevOutput = this.layerOutputs[i - 1];
          if (prevOutput) {
            this.graph.outputs.push(
              new ValueInfo('currentInput', ['batch_size', -1, -1, -1], 'float32'),
            );
          }
        }
      }
    } else {
      this.graph.outputs.push(new ValueInfo('currentInput', ['batch_size', -1, -1, -1], 'float32'));
    }
  }

  /**
   * Reads `count` float weights from the weight buffer, padding with zeros if exhausted.
   * @param count Number of float values to read.
   * @returns `Float32Array` of length `count`.
   */
  private readWeights(count: number): Float32Array {
    const available = Math.min(count, this.weights.length - this.weightOffset);
    let slice: Float32Array;
    if (available > 0) {
      slice = this.weights.slice(this.weightOffset, this.weightOffset + available);
    } else {
      slice = new Float32Array(count);
    }
    if (slice.length < count) {
      const padded = new Float32Array(count);
      padded.set(slice);
      slice = padded;
    }
    this.weightOffset += count;
    return slice;
  }

  /**
   * Maps a single Darknet execution section to ONNX nodes.
   * @param layer Darknet section descriptor.
   * @param layerIdx Zero-based execution layer index.
   * @param input Name of the current input tensor.
   * @param inChannels Number of input channels.
   * @returns Emitted ONNX nodes for this section.
   */
  private mapLayer(
    layer: DarknetSection,
    layerIdx: number,
    input: string,
    inChannels: number,
  ): Node[] {
    const nodes: Node[] = [];
    const layerName = `layer_${layerIdx}_${layer.type}`;

    switch (layer.type) {
      case 'convolutional': {
        const filters = layer.filters || 1;
        const size = layer.size || 1;
        const stride = layer.stride || 1;
        const pad = layer.pad ? Math.floor(size / 2) : 0;
        const batchNormalize = layer.batch_normalize === 1;

        const weightCount = filters * inChannels * size * size;

        if (batchNormalize) {
          this.readWeights(filters);
          this.readWeights(filters);
          this.readWeights(filters);
          this.readWeights(filters);
          this.readWeights(weightCount);
        } else {
          this.readWeights(filters);
          this.readWeights(weightCount);
        }

        const convNode = new Node(
          'Conv',
          batchNormalize ? [input, `${layerName}_W`] : [input, `${layerName}_W`, `${layerName}_B`],
          [`${layerName}_conv_out`],
          {
            kernel_shape: new Attribute('kernel_shape', 'INTS', [size, size]),
            strides: new Attribute('strides', 'INTS', [stride, stride]),
            pads: new Attribute('pads', 'INTS', [pad, pad, pad, pad]),
          },
          `${layerName}_conv`,
        );

        nodes.push(convNode);
        let nextInput = `${layerName}_conv_out`;

        if (batchNormalize) {
          const t_scale = new Tensor(`${layerName}_scale`, [filters], 'float32', true);
          t_scale.data = new Float32Array(this.readWeights(filters));
          this.graph.tensors[`${layerName}_scale`] = t_scale;
          this.graph.initializers.push(`${layerName}_scale`);

          const t_B = new Tensor(`${layerName}_B`, [filters], 'float32', true);
          t_B.data = new Float32Array(this.readWeights(filters));
          this.graph.tensors[`${layerName}_B`] = t_B;
          this.graph.initializers.push(`${layerName}_B`);

          const t_mean = new Tensor(`${layerName}_mean`, [filters], 'float32', true);
          t_mean.data = new Float32Array(this.readWeights(filters));
          this.graph.tensors[`${layerName}_mean`] = t_mean;
          this.graph.initializers.push(`${layerName}_mean`);

          const t_var = new Tensor(`${layerName}_var`, [filters], 'float32', true);
          t_var.data = new Float32Array(this.readWeights(filters));
          this.graph.tensors[`${layerName}_var`] = t_var;
          this.graph.initializers.push(`${layerName}_var`);

          const bnNode = new Node(
            'BatchNormalization',
            [
              nextInput,
              `${layerName}_scale`,
              `${layerName}_B`,
              `${layerName}_mean`,
              `${layerName}_var`,
            ],
            [`${layerName}_bn_out`],
            { epsilon: new Attribute('epsilon', 'FLOAT', 1e-5) },
            `${layerName}_bn`,
          );
          nodes.push(bnNode);
          nextInput = `${layerName}_bn_out`;
        }

        if (layer.activation === 'leaky') {
          const reluNode = new Node(
            'LeakyRelu',
            [nextInput],
            [`${layerName}_out`],
            { alpha: new Attribute('alpha', 'FLOAT', 0.1) },
            `${layerName}_relu`,
          );
          nodes.push(reluNode);
        } else if (layer.activation === 'mish') {
          const mishNode = new Node(
            'Mish',
            [nextInput],
            [`${layerName}_out`],
            {},
            `${layerName}_mish`,
          );
          nodes.push(mishNode);
        } else if (layer.activation === 'swish') {
          const sigNode = new Node(
            'Sigmoid',
            [nextInput],
            [`${layerName}_sig_out`],
            {},
            `${layerName}_sig`,
          );
          const mulNode = new Node(
            'Mul',
            [nextInput, `${layerName}_sig_out`],
            [`${layerName}_out`],
            {},
            `${layerName}_swish`,
          );
          nodes.push(sigNode);
          nodes.push(mulNode);
        } else {
          const lastNode = nodes[nodes.length - 1];
          if (lastNode && lastNode.outputs.length > 0) {
            lastNode.outputs[0] = `${layerName}_out`;
          }
        }

        this.channelsOutput[layerIdx] = filters;
        break;
      }

      case 'maxpool': {
        const size = layer.size || 2;
        const stride = layer.stride || 2;
        nodes.push(
          new Node(
            'MaxPool',
            [input],
            [`${layerName}_out`],
            {
              kernel_shape: new Attribute('kernel_shape', 'INTS', [size, size]),
              strides: new Attribute('strides', 'INTS', [stride, stride]),
            },
            layerName,
          ),
        );
        this.channelsOutput[layerIdx] = inChannels;
        break;
      }
      case 'avgpool': {
        nodes.push(new Node('AveragePool', [input], [`${layerName}_out`], {}, layerName));
        this.channelsOutput[layerIdx] = inChannels;
        break;
      }

      case 'connected': {
        const outputSize = layer.output || 1;
        const weightCount = outputSize * inChannels;

        const biases = this.readWeights(outputSize);
        const gemmWeights = this.readWeights(weightCount);

        const t_W = new Tensor(`${layerName}_W`, [outputSize, inChannels], 'float32', true);
        t_W.data = new Float32Array(gemmWeights);
        this.graph.tensors[`${layerName}_W`] = t_W;
        this.graph.initializers.push(`${layerName}_W`);

        const t_B = new Tensor(`${layerName}_B`, [outputSize], 'float32', true);
        t_B.data = new Float32Array(biases);
        this.graph.tensors[`${layerName}_B`] = t_B;
        this.graph.initializers.push(`${layerName}_B`);

        nodes.push(
          new Node(
            'Gemm',
            [input, `${layerName}_W`, `${layerName}_B`],
            [`${layerName}_out`],
            { transB: new Attribute('transB', 'INT', 1) },
            layerName,
          ),
        );

        this.channelsOutput[layerIdx] = outputSize;
        break;
      }
      case 'shortcut': {
        const fromIndex = layer.from ?? -1;
        const targetIdx = fromIndex < 0 ? layerIdx + fromIndex : fromIndex;
        const targetInput = this.layerOutputs[targetIdx] ?? '';
        nodes.push(new Node('Add', [input, targetInput], [`${layerName}_out`], {}, layerName));
        break;
      }
      case 'route': {
        const rawRoute = layer.layers ?? 0;
        const layersList = Array.isArray(rawRoute) ? rawRoute : [rawRoute];
        const inputs = layersList.map((l: number) => {
          const targetIdx = l < 0 ? layerIdx + l : l;
          return this.layerOutputs[targetIdx] ?? '';
        });

        if (inputs.length === 1) {
          nodes.push(
            new Node('Identity', [inputs[0] as string], [`${layerName}_out`], {}, layerName),
          );
        } else {
          nodes.push(
            new Node(
              'Concat',
              inputs,
              [`${layerName}_out`],
              { axis: new Attribute('axis', 'INT', 1) },
              layerName,
            ),
          );
        }
        break;
      }
      case 'upsample': {
        const stride = layer.stride || 2;
        const scalesName = `${layerName}_scales`;

        const t_scales = new Tensor(scalesName, [4], 'float32', true);
        t_scales.data = new Float32Array([1, 1, stride, stride]);
        this.graph.tensors[scalesName] = t_scales;
        this.graph.initializers.push(scalesName);

        nodes.push(
          new Node(
            'Resize',
            [input, '', scalesName],
            [`${layerName}_out`],
            { mode: new Attribute('mode', 'STRING', 'nearest') },
            layerName,
          ),
        );
        this.channelsOutput[layerIdx] = inChannels;
        break;
      }
      case 'yolo':
      case 'region':
        break;
    }

    return nodes;
  }
}
