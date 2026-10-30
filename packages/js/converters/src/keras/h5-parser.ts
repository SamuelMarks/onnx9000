/**
 * @fileoverview h5-parser.ts
 * Provides h5-parser functionality for the converters package.
 */
import { ConversionError } from '@onnx9000/core';
import { type Dataset, type Group, File as Hdf5File } from 'jsfive';
import type { JsonObject } from './tfjs-parser.js';

/** Interface `KerasH5Model` definition. */
export interface KerasH5Model {
  /** Property `modelConfig`. */
  modelConfig: JsonObject;
  /** Property `kerasVersion`. */
  kerasVersion: string;
  /** Property `backend`. */
  backend: string;
  /** Property `weights`. */
  weights: Record<string, H5Weight>;
}

/** Interface `H5Weight` definition. */
export interface H5Weight {
  /** Property `name`. */
  name: string;
  /** Property `shape`. */
  shape: number[];
  /** Property `data`. */
  data: Float32Array | Int32Array | Uint8Array | Float64Array;
}

/**
 * Executes `parseKerasH5`.
 * @param buffer Parameter `buffer`.
 * @returns Result of `parseKerasH5`.
 */
export function parseKerasH5(buffer: ArrayBuffer): KerasH5Model {
  const f = new Hdf5File(buffer, 'model.h5');

  // Root attrs
  const rootAttrs = (f as never as { attrs: Record<string, string | Uint8Array> }).attrs;
  let modelConfigJson = '{}';

  if (rootAttrs.model_config) {
    const configRaw = rootAttrs.model_config;
    modelConfigJson =
      typeof configRaw === 'string' ? configRaw : new TextDecoder().decode(configRaw);
  } else {
    throw new ConversionError('HDF5 file does not contain a Keras model_config attribute.');
  }

  const modelConfig = JSON.parse(modelConfigJson) as JsonObject;

  let kerasVersion = 'unknown';
  if (rootAttrs.keras_version) {
    const vRaw = rootAttrs.keras_version;
    kerasVersion = typeof vRaw === 'string' ? vRaw : new TextDecoder().decode(vRaw);
  }

  let backend = 'unknown';
  if (rootAttrs.backend) {
    const bRaw = rootAttrs.backend;
    backend = typeof bRaw === 'string' ? bRaw : new TextDecoder().decode(bRaw);
  }

  const weights: Record<string, H5Weight> = {};

  // Keras typically stores weights in /model_weights or root
  let weightsGroup: Group | undefined;
  try {
    const potentialGroup = f.get('model_weights');
    if (potentialGroup && 'keys' in potentialGroup) {
      weightsGroup = potentialGroup as Group;
    }
  } catch {
    // sometimes they are just in root if it's a weights-only file
    weightsGroup = f as never as Group;
  }

  if (weightsGroup?.keys) {
    for (const layerName of weightsGroup.keys) {
      let layerGroup: Group;
      try {
        layerGroup = weightsGroup.get(layerName) as Group;
      } catch {
        continue;
      }
      if (!layerGroup || !('keys' in layerGroup)) continue;

      const weightNamesRaw = layerGroup.attrs.weight_names;
      let weightNames: string[] = [];
      if (weightNamesRaw) {
        if (Array.isArray(weightNamesRaw)) {
          weightNames = weightNamesRaw.map((v) => (typeof v === 'string' ? v : String(v)));
        } else if (typeof weightNamesRaw === 'string') {
          weightNames = [weightNamesRaw];
        }
      } else {
        weightNames = layerGroup.keys; // fallback
      }

      for (const wName of weightNames) {
        // The dataset path is typically layerName/wName, but Keras can nest it
        // e.g., model_weights/dense_1/dense_1/kernel:0
        let wDataset: Dataset;
        try {
          // Keras sometimes nests layerName -> layerName -> weight
          // Or layerName -> weight
          let ds: object = layerGroup.get(wName);
          if (ds && 'keys' in (ds as Group)) {
            // It's a group, try to get the actual weight from inside it (using wName again or just the first key)
            const innerGroup = ds as Group;
            if (innerGroup.keys.length > 0) {
              ds = innerGroup.get(innerGroup.keys[0] as string);
            }
          }
          wDataset = ds as Dataset;
        } catch {
          continue;
        }

        if (wDataset && wDataset.shape !== undefined && wDataset.value !== undefined) {
          weights[wName] = {
            name: wName,
            shape: wDataset.shape,
            data: wDataset.value,
          };
        }
      }
    }
  }

  return {
    modelConfig,
    kerasVersion,
    backend,
    weights,
  };
}
