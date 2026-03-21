import { parseTFJSModel, TFJSModel } from './tfjs-parser.js';
import { extractKerasTopology, KerasModelTopology } from './keras-ast.js';
import { emitConv, emitSeparableConv } from './emitters-conv.js';
import { emitDense, emitActivation, emitIdentity } from './emitters.js';
import { emitPool, emitGlobalPool } from './emitters-pool.js';
import { LayoutOptimizer } from './layout-optimizer.js';

export class Keras2OnnxConverter {
  private topology: KerasModelTopology;
  private nodes: any[] = [];
  private layoutOptimizer = new LayoutOptimizer();

  constructor(modelJson: string) {
    const model = parseTFJSModel(modelJson);
    this.topology = extractKerasTopology(model.modelTopology);
  }

  public convert(): Uint8Array {
    // High level translation loop
    const entries = Array.from(this.topology.layers.entries());
    for (let i = 0; i < entries.length; i++) {
      const [layerName, layer] = entries[i]!;
      this.translateLayer(layerName, layer);
    }

    // Optimize layout
    this.nodes = this.layoutOptimizer.optimize(this.nodes);

    // Here we would normally build an ONNX ModelProto using pure JS protobuf encoder
    // returning an empty Uint8Array for now as the ONNX AST serializer isn't fully available in this snippet
    return new Uint8Array(0);
  }

  private translateLayer(layerName: string, layer: any) {
    const className = layer.className;
    const config = layer.config;

    const inputNames =
      layer.inboundNodes.length > 0 ? layer.inboundNodes[0] : [layerName + '_input'];
    const inName = inputNames[0]; // simplistic single input

    switch (className) {
      case 'InputLayer':
        // Handled implicitly by graph inputs
        break;
      case 'Dense': {
        const units = config.units as number;
        const activation = (config.activation as string) || 'linear';
        const useBias = config.use_bias !== false;

        this.nodes.push(
          ...emitDense(
            inName,
            layerName,
            layerName + '_weights',
            useBias ? layerName + '_bias' : undefined,
            activation,
            layerName,
          ),
        );
        break;
      }
      case 'Activation': {
        const activation = config.activation as string;
        this.nodes.push(...emitActivation(activation, inName, layerName, layerName));
        break;
      }
      case 'Conv2D': {
        const activation = (config.activation as string) || 'linear';
        const padding = (config.padding as string) === 'same' ? 'same' : 'valid';
        const strides = config.strides as number[];
        const dilation_rate = config.dilation_rate as number[];
        const kernel_size = config.kernel_size as number[];

        this.nodes.push(
          ...emitConv(
            'Conv',
            inName,
            layerName,
            layerName + '_kernel',
            layerName + '_bias',
            layerName,
            {
              activation,
              padding,
              strides,
              dilations: dilation_rate,
              kernelShape: kernel_size,
            },
          ),
        );
        break;
      }
      case 'MaxPooling2D':
      case 'AveragePooling2D': {
        const isMax = className.startsWith('Max');
        const padding = (config.padding as string) === 'same' ? 'same' : 'valid';
        const pool_size = config.pool_size as number[];
        const strides = config.strides as number[];

        this.nodes.push(
          ...emitPool(isMax ? 'Max' : 'Average', inName, layerName, layerName, {
            padding,
            poolSize: pool_size,
            strides,
          }),
        );
        break;
      }
      case 'GlobalAveragePooling2D':
      case 'GlobalMaxPooling2D': {
        const isMax = className.startsWith('GlobalMax');
        const keepDims = config.keepdims === true;
        this.nodes.push(
          ...emitGlobalPool(isMax ? 'Max' : 'Average', inName, layerName, layerName, { keepDims }),
        );
        break;
      }
      case 'Flatten':
      case 'Dropout':
        // Emitting Identity for unsupported layers to keep graph contiguous
        this.nodes.push(...emitIdentity(inName, layerName, layerName));
        break;
      default:
        // Unhandled layer type during validation
        break;
    }
  }
}
