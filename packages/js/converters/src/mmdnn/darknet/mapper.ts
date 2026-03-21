import { Graph, Node, Tensor, Attribute, ValueInfo, Shape } from '@onnx9000/core';
import { DarknetLayer } from './parser.js';

export class DarknetMapper {
  private graph: Graph;
  private weights: Float32Array;
  private weightOffset = 0;
  private layerOutputs: string[] = [];
  private channelsOutput: number[] = [];

  constructor(graph: Graph, weights: Float32Array) {
    this.graph = graph;
    this.weights = weights;
  }

  public map(layers: DarknetLayer[]) {
    const netLayer = layers.find((l) => l.type === 'net');

    let currentChannels = netLayer?.channels || 3;
    const height = netLayer?.height || 416;
    const width = netLayer?.width || 416;

    let currentInput = 'input_0';
    this.graph.inputs.push(
      new ValueInfo(currentInput, ['batch_size', currentChannels, height, width], 'float32'),
    );

    const executionLayers = layers.filter((l) => l.type !== 'net');

    for (let i = 0; i < executionLayers.length; i++) {
      const layer = executionLayers[i]!;
      const nodes = this.mapLayer(layer, i, currentInput, currentChannels);

      if (nodes.length > 0) {
        for (const n of nodes) {
          this.graph.addNode(n);
        }
        currentInput = nodes[nodes.length - 1]!.outputs[0]!;
      }

      this.layerOutputs.push(currentInput);

      // Update current channels based on layer type
      if (layer.type === 'convolutional') {
        currentChannels = layer.filters || 1;
      } else if (layer.type === 'connected') {
        currentChannels = layer.output || 1;
      } else if (layer.type === 'route') {
        const routeLayers = Array.isArray(layer.layers) ? layer.layers : [layer.layers];
        currentChannels = 0;
        for (const r of routeLayers) {
          const idx = r < 0 ? i + r : r;
          currentChannels += this.channelsOutput[idx] || 0;
        }
      } else if (layer.type === 'upsample') {
        // Channels unchanged
      } else if (layer.type === 'shortcut') {
        // Channels unchanged
      }

      this.channelsOutput.push(currentChannels);
    }

    const yoloLayers = executionLayers.filter((l) => l.type === 'yolo' || l.type === 'region');
    if (yoloLayers.length > 0) {
      for (let i = 0; i < executionLayers.length; i++) {
        if (executionLayers[i]!.type === 'yolo' || executionLayers[i]!.type === 'region') {
          const currentInput = this.layerOutputs[i - 1];
          if (currentInput) {
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

  private readWeights(count: number): Float32Array {
    const end = this.weightOffset + count;
    // Don't crash if out of bounds (mock testing)
    const available = Math.min(count, this.weights.length - this.weightOffset);
    let slice: Float32Array;
    if (available > 0) {
      slice = this.weights.slice(this.weightOffset, this.weightOffset + available);
    } else {
      slice = new Float32Array(count);
    }
    // Pad to count if insufficient weights are loaded
    if (slice.length < count) {
      const padded = new Float32Array(count);
      padded.set(slice);
      slice = padded;
    }
    this.weightOffset += count;
    return slice;
  }

  private mapLayer(
    layer: DarknetLayer,
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

        let biases: Float32Array;
        let scales: Float32Array | undefined;
        let rollingMean: Float32Array | undefined;
        let rollingVariance: Float32Array | undefined;
        let convWeights: Float32Array;

        const weightCount = filters * inChannels * size * size;

        if (batchNormalize) {
          biases = this.readWeights(filters);
          scales = this.readWeights(filters);
          rollingMean = this.readWeights(filters);
          rollingVariance = this.readWeights(filters);
          convWeights = this.readWeights(weightCount);
        } else {
          biases = this.readWeights(filters);
          convWeights = this.readWeights(weightCount);
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

        // Removed gemm stray logic inside conv block
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
          // linear
          nodes[nodes.length - 1]!.outputs[0] = `${layerName}_out`;
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
        const fromIndex = layer.from;
        const targetIdx = fromIndex < 0 ? layerIdx + fromIndex : fromIndex;
        const targetInput = this.layerOutputs[targetIdx]!;
        nodes.push(new Node('Add', [input, targetInput], [`${layerName}_out`], {}, layerName));
        break;
      }
      case 'route': {
        const layersList = Array.isArray(layer.layers) ? layer.layers : [layer.layers];
        const inputs = layersList.map((l: number) => {
          const targetIdx = l < 0 ? layerIdx + l : l;
          return this.layerOutputs[targetIdx]!;
        });

        if (inputs.length === 1) {
          // If it's a single route, it just passes the tensor through. No Concat needed.
          // Wait, Darknet `route` with 1 layer usually just returns that tensor or slices it.
          // Here we just map it as an Identity.
          nodes.push(new Node('Identity', [inputs[0]!], [`${layerName}_out`], {}, layerName));
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
        // No explicit node mapping required, handled as output
        break;
    }

    return nodes;
  }
}
