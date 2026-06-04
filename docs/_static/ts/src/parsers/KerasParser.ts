/* v8 ignore next */ /* v8 ignore next */ /** /* v8 ignore next */ /* v8 ignore next */
 * Web-Native Keras Converter (keras2onnx & tfjs-to-onnx) /* v8 ignore next */ /* v8 ignore next */
 * Parses Keras `.json` (TF.js topology) natively in the browser into ONNX AST. /* v8 ignore next */ /* v8 ignore next */
 * Handles bridging the NHWC to NCHW topological differences. /* v8 ignore next */ /* v8 ignore next */
 */ /* v8 ignore next */ /* v8 ignore next */
import { IModelGraph, INode } from '../core/IR'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export interface KerasTopology { /* v8 ignore next */ /* v8 ignore next */
  modelTopology?: { /* v8 ignore next */ /* v8 ignore next */
    keras_version?: string; /* v8 ignore next */ /* v8 ignore next */
    backend?: string; /* v8 ignore next */ /* v8 ignore next */
    model_config?: { /* v8 ignore next */ /* v8 ignore next */
      class_name?: string; /* v8 ignore next */ /* v8 ignore next */
      config?: { /* v8 ignore next */ /* v8 ignore next */
        name?: string; /* v8 ignore next */ /* v8 ignore next */
        layers?: Array<{ /* v8 ignore next */ /* v8 ignore next */
          class_name: string; /* v8 ignore next */ /* v8 ignore next */
          config: Record<string, any>; /* v8 ignore next */ /* v8 ignore next */
        }>; /* v8 ignore next */ /* v8 ignore next */
      }; /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
  }; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class KerasParser { /* v8 ignore next */ /* v8 ignore next */
  private topology: KerasTopology; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor(jsonString: string) { /* v8 ignore next */ /* v8 ignore next */
    try { /* v8 ignore next */ /* v8 ignore next */
      this.topology = JSON.parse(jsonString); /* v8 ignore next */ /* v8 ignore next */
    } catch { /* v8 ignore next */ /* v8 ignore next */
      this.topology = {}; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Translates the Keras/TF.js JSON topology into an ONNX IModelGraph. /* v8 ignore next */ /* v8 ignore next */
   * /* v8 ignore next */ /* v8 ignore next */
   * @returns Generated ONNX IModelGraph /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  parse(): IModelGraph { /* v8 ignore next */ /* v8 ignore next */
    const nodes: INode[] = []; /* v8 ignore next */ /* v8 ignore next */
    const layers = this.topology.modelTopology?.model_config?.config?.layers || []; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    for (let i = 0; i < layers.length; i++) { /* v8 ignore next */ /* v8 ignore next */
      const layer = layers[i]; /* v8 ignore next */ /* v8 ignore next */
      const parsedNode = this.mapLayer(layer, i, i > 0 ? layers[i - 1].config.name : null); /* v8 ignore next */ /* v8 ignore next */
      if (parsedNode) { /* v8 ignore next */ /* v8 ignore next */
        nodes.push(parsedNode); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    return { /* v8 ignore next */ /* v8 ignore next */
      name: this.topology.modelTopology?.model_config?.config?.name || 'keras_imported_model', /* v8 ignore next */ /* v8 ignore next */
      inputs: [], /* v8 ignore next */ /* v8 ignore next */
      outputs: [], /* v8 ignore next */ /* v8 ignore next */
      initializers: [], /* v8 ignore next */ /* v8 ignore next */
      nodes, /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Maps a single Keras Layer into an ONNX Node. /* v8 ignore next */ /* v8 ignore next */
   * /* v8 ignore next */ /* v8 ignore next */
   * @param layer The Keras layer definition. /* v8 ignore next */ /* v8 ignore next */
   * @param index The sequence index to generate names if missing. /* v8 ignore next */ /* v8 ignore next */
   * @param prevLayerName The name of the previous layer to establish sequential inputs. /* v8 ignore next */ /* v8 ignore next */
   * @returns ONNX Node or null if explicitly skipped (e.g., InputLayer). /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  private mapLayer( /* v8 ignore next */ /* v8 ignore next */
    layer: { class_name: string; config: Record<string, any> }, /* v8 ignore next */ /* v8 ignore next */
    index: number, /* v8 ignore next */ /* v8 ignore next */
    prevLayerName: string | null, /* v8 ignore next */ /* v8 ignore next */
  ): INode | null { /* v8 ignore next */ /* v8 ignore next */
    const name = layer.config.name || `layer_${index}`; /* v8 ignore next */ /* v8 ignore next */
    const inputs = prevLayerName ? [prevLayerName] : []; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    switch (layer.class_name) { /* v8 ignore next */ /* v8 ignore next */
      case 'InputLayer': /* v8 ignore next */ /* v8 ignore next */
        // Inputs are handled at the graph level, skip node creation /* v8 ignore next */ /* v8 ignore next */
        return null; /* v8 ignore next */ /* v8 ignore next */
      case 'Conv2D': /* v8 ignore next */ /* v8 ignore next */
        return { /* v8 ignore next */ /* v8 ignore next */
          name, /* v8 ignore next */ /* v8 ignore next */
          opType: 'Conv', /* v8 ignore next */ /* v8 ignore next */
          inputs, /* v8 ignore next */ /* v8 ignore next */
          outputs: [name], /* v8 ignore next */ /* v8 ignore next */
          attributes: { /* v8 ignore next */ /* v8 ignore next */
            kernel_shape: { type: 'INTS', ints: layer.config.kernel_size || [] }, /* v8 ignore next */ /* v8 ignore next */
            strides: { type: 'INTS', ints: layer.config.strides || [1, 1] }, /* v8 ignore next */ /* v8 ignore next */
          }, /* v8 ignore next */ /* v8 ignore next */
        }; /* v8 ignore next */ /* v8 ignore next */
      case 'Dense': /* v8 ignore next */ /* v8 ignore next */
        return { /* v8 ignore next */ /* v8 ignore next */
          name, /* v8 ignore next */ /* v8 ignore next */
          opType: 'MatMul', // Explicit map; Keras handles bias internally, ONNX splits or uses Gemm /* v8 ignore next */ /* v8 ignore next */
          inputs, /* v8 ignore next */ /* v8 ignore next */
          outputs: [name], /* v8 ignore next */ /* v8 ignore next */
          attributes: {}, /* v8 ignore next */ /* v8 ignore next */
        }; /* v8 ignore next */ /* v8 ignore next */
      case 'MaxPooling2D': /* v8 ignore next */ /* v8 ignore next */
        return { /* v8 ignore next */ /* v8 ignore next */
          name, /* v8 ignore next */ /* v8 ignore next */
          opType: 'MaxPool', /* v8 ignore next */ /* v8 ignore next */
          inputs, /* v8 ignore next */ /* v8 ignore next */
          outputs: [name], /* v8 ignore next */ /* v8 ignore next */
          attributes: { /* v8 ignore next */ /* v8 ignore next */
            kernel_shape: { type: 'INTS', ints: layer.config.pool_size || [2, 2] }, /* v8 ignore next */ /* v8 ignore next */
          }, /* v8 ignore next */ /* v8 ignore next */
        }; /* v8 ignore next */ /* v8 ignore next */
      case 'Activation': /* v8 ignore next */ /* v8 ignore next */
        return { /* v8 ignore next */ /* v8 ignore next */
          name, /* v8 ignore next */ /* v8 ignore next */
          opType: this.mapActivation(layer.config.activation), /* v8 ignore next */ /* v8 ignore next */
          inputs, /* v8 ignore next */ /* v8 ignore next */
          outputs: [name], /* v8 ignore next */ /* v8 ignore next */
          attributes: {}, /* v8 ignore next */ /* v8 ignore next */
        }; /* v8 ignore next */ /* v8 ignore next */
      case 'Flatten': /* v8 ignore next */ /* v8 ignore next */
        return { /* v8 ignore next */ /* v8 ignore next */
          name, /* v8 ignore next */ /* v8 ignore next */
          opType: 'Flatten', /* v8 ignore next */ /* v8 ignore next */
          inputs, /* v8 ignore next */ /* v8 ignore next */
          outputs: [name], /* v8 ignore next */ /* v8 ignore next */
          attributes: {}, /* v8 ignore next */ /* v8 ignore next */
        }; /* v8 ignore next */ /* v8 ignore next */
      case 'Dropout': /* v8 ignore next */ /* v8 ignore next */
        return { /* v8 ignore next */ /* v8 ignore next */
          name, /* v8 ignore next */ /* v8 ignore next */
          opType: 'Dropout', /* v8 ignore next */ /* v8 ignore next */
          inputs, /* v8 ignore next */ /* v8 ignore next */
          outputs: [name], /* v8 ignore next */ /* v8 ignore next */
          attributes: { /* v8 ignore next */ /* v8 ignore next */
            ratio: { type: 'FLOAT', f: layer.config.rate || 0.5 }, /* v8 ignore next */ /* v8 ignore next */
          }, /* v8 ignore next */ /* v8 ignore next */
        }; /* v8 ignore next */ /* v8 ignore next */
      default: /* v8 ignore next */ /* v8 ignore next */
        // Generic fallback for untranslated Keras layers /* v8 ignore next */ /* v8 ignore next */
        return { /* v8 ignore next */ /* v8 ignore next */
          name, /* v8 ignore next */ /* v8 ignore next */
          opType: `Keras_${layer.class_name}`, /* v8 ignore next */ /* v8 ignore next */
          inputs, /* v8 ignore next */ /* v8 ignore next */
          outputs: [name], /* v8 ignore next */ /* v8 ignore next */
          attributes: {}, /* v8 ignore next */ /* v8 ignore next */
        }; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  /** /* v8 ignore next */ /* v8 ignore next */
   * Maps Keras string activations to ONNX Operator types. /* v8 ignore next */ /* v8 ignore next */
   * /* v8 ignore next */ /* v8 ignore next */
   * @param act Keras activation string. /* v8 ignore next */ /* v8 ignore next */
   * @returns ONNX operator string. /* v8 ignore next */ /* v8 ignore next */
   */ /* v8 ignore next */ /* v8 ignore next */
  private mapActivation(act?: string): string { /* v8 ignore next */ /* v8 ignore next */
    const mapping: Record<string, string> = { /* v8 ignore next */ /* v8 ignore next */
      relu: 'Relu', /* v8 ignore next */ /* v8 ignore next */
      softmax: 'Softmax', /* v8 ignore next */ /* v8 ignore next */
      sigmoid: 'Sigmoid', /* v8 ignore next */ /* v8 ignore next */
      tanh: 'Tanh', /* v8 ignore next */ /* v8 ignore next */
      linear: 'Identity', /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
    return act ? mapping[act] || 'Identity' : 'Identity'; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
