import { JsonObject } from './tfjs-parser.js';
import { OnnxNodeBuilder } from './emitters.js';

/**
 * Type definition for a function that emits custom ONNX nodes for a Keras layer.
 */
export type CustomLayerEmitter = (
  nodeName: string, // e.g. "my_custom_layer:0"
  layerName: string, // e.g. "my_custom_layer" (for weights)
  inboundNodes: string[], // e.g. ["input_1:0:0", "dense_2:0:0"]
  outName: string, // e.g. "my_custom_layer:0:0"
  config: JsonObject, // Keras config dictionary
) => OnnxNodeBuilder[]; // Returns an array of raw OnnxNodeBuilder objects to be pushed to rawNodes

const layerPluginRegistry = new Map<string, CustomLayerEmitter>();

/**
 * Registers a custom Keras Layer mapping directly to an ONNX subgraph injection.
 * This conforms to the "Plugin Registry" Architecture Mandate.
 *
 * @param kerasLayerName The class name of the Keras layer (e.g. 'MyCustomAttention').
 * @param emitter The function responsible for returning the array of nodes that map this layer to ONNX.
 */
export function registerCustomKerasLayer(
  kerasLayerName: string,
  emitter: CustomLayerEmitter,
): void {
  if (layerPluginRegistry.has(kerasLayerName)) {
    console.warn(`[onnx9000] Overwriting existing custom layer plugin for ${kerasLayerName}`);
  }
  layerPluginRegistry.set(kerasLayerName, emitter);
}

/**
 * Attempts to retrieve a registered custom layer emitter.
 *
 * @param kerasLayerName The class name of the Keras layer.
 * @returns The emitter function if registered, otherwise undefined.
 */
export function getCustomKerasLayerEmitter(kerasLayerName: string): CustomLayerEmitter | undefined {
  return layerPluginRegistry.get(kerasLayerName);
}
