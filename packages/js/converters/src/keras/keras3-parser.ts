/* eslint-disable */
// @ts-nocheck
import { unzipSync } from 'fflate';
import { JsonObject } from './tfjs-parser.js';

export interface Keras3Model {
  config: JsonObject;
  weights: Record<string, Uint8Array>;
  metadata: JsonObject;
}

export function parseKeras3Zip(buffer: Uint8Array): Keras3Model {
  const unzipped = unzipSync(buffer);

  let config: JsonObject | undefined;
  let metadata: JsonObject | undefined;
  const weights: Record<string, Uint8Array> = {};

  for (const [filename, fileData] of Object.entries(unzipped)) {
    if (filename.endsWith('config.json')) {
      const text = new TextDecoder().decode(fileData);
      config = JSON.parse(text) as JsonObject;
    } else if (filename.endsWith('metadata.json')) {
      const text = new TextDecoder().decode(fileData);
      metadata = JSON.parse(text) as JsonObject;
    } else if (filename.endsWith('.weights.h5')) {
      // Keras 3 still bundles weights in an H5 file inside the zip.
      // We store the raw buffer for the H5 parser to handle later.
      weights[filename] = fileData;
    } else if (filename.includes('weights/')) {
      // Some formats might use safetensors or flat bins
      weights[filename] = fileData;
    }
  }

  if (config === undefined) {
    throw new Error('Invalid Keras 3 format: missing config.json');
  }

  return {
    config,
    metadata: metadata !== undefined ? metadata : {},
    weights,
  };
}
