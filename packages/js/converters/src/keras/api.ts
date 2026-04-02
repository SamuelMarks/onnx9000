/* eslint-disable */
// @ts-nocheck
import { parseTFJSModel } from './tfjs-parser.js';
import { parseKerasH5 } from './h5-parser.js';
import { Keras2OnnxConverter } from './index.js';

/**
 * Converts a Keras model to ONNX format.
 * @param modelData The Keras model topology as a JSON string (TF.js format) or an HDF5 ArrayBuffer.
 * @param weightsBin Optional ArrayBuffer containing weights.
 * @returns A Promise resolving to the serialized ONNX ModelProto as a Uint8Array.
 */
export async function keras2onnx(
  modelData: string | ArrayBuffer,
  weightsBin?: ArrayBuffer,
): Promise<Uint8Array> {
  if (typeof modelData === 'string') {
    // TF.js model.json string
    const converter = new Keras2OnnxConverter(modelData);
    // Note: weightsBin is omitted in this simplified signature for now
    return converter.convert();
  } else {
    // HDF5 ArrayBuffer
    const h5model = parseKerasH5(modelData);
    // Extract json
    const converter = new Keras2OnnxConverter(JSON.stringify(h5model.modelConfig));
    return converter.convert();
  }
}
