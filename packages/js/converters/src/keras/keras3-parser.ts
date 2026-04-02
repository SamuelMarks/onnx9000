import { unzipSync } from 'fflate';
import { JsonObject } from './tfjs-parser.js';

/**
 * Represents the extracted artifacts from a Keras 3 (.keras) archive.
 */
export interface Keras3Model {
  config: JsonObject;
  metadata: JsonObject;
  weightsH5?: Uint8Array;
  weightsSafetensors?: Uint8Array;
}

/**
 * Parses a Keras 3 (.keras) archive utilizing JS zip extraction to read the necessary JSON files and binary weight data.
 *
 * @param buffer The raw binary buffer of the .keras archive.
 * @returns A structured object containing the parsed config, metadata, and optional weight buffers.
 * @throws Error if the provided buffer is not a valid Keras 3 archive missing config.json.
 */
export function parseKeras3Zip(buffer: Uint8Array): Keras3Model {
  const unzipped = unzipSync(buffer);

  let config: JsonObject | undefined = undefined;
  let metadata: JsonObject | undefined = undefined;
  let weightsH5: Uint8Array | undefined = undefined;
  let weightsSafetensors: Uint8Array | undefined = undefined;

  for (const entry of Object.entries(unzipped)) {
    const filename = entry[0];
    const fileData = entry[1];

    if (filename === 'config.json' || filename.endsWith('/config.json')) {
      const text = new TextDecoder().decode(fileData);
      // We safely cast the output of JSON.parse to JsonObject to avoid any/unknown
      config = JSON.parse(text) as JsonObject;
    } else if (filename === 'metadata.json' || filename.endsWith('/metadata.json')) {
      const text = new TextDecoder().decode(fileData);
      metadata = JSON.parse(text) as JsonObject;
    } else if (filename === 'model.weights.h5' || filename.endsWith('/model.weights.h5')) {
      weightsH5 = fileData;
    } else if (
      filename === 'model.weights.safetensors' ||
      filename.endsWith('/model.weights.safetensors')
    ) {
      weightsSafetensors = fileData;
    }
  }

  if (config === undefined) {
    throw new Error('Invalid Keras 3 format: missing config.json');
  }

  const result: Keras3Model = {
    config,
    metadata: metadata !== undefined ? metadata : {},
  };

  if (weightsH5 !== undefined) {
    result.weightsH5 = weightsH5;
  }
  if (weightsSafetensors !== undefined) {
    result.weightsSafetensors = weightsSafetensors;
  }

  return result;
}
