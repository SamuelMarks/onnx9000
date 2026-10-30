/**
 * @fileoverview savedmodel-parser module for converters.
 */
/**
 * Extracted raw representations from a TF SavedModel directory
 */
import { ConversionError } from '@onnx9000/core';
/** Interface `SavedModel` definition. */
export interface SavedModel {
  /** Property `savedModelPb`. */
  savedModelPb: Uint8Array;
  /** Property `variablesIndex`. */
  variablesIndex?: Uint8Array;
  /** Property `variablesData`. */
  variablesData: Uint8Array[];
}

/**
 * Parses a TF SavedModel directory structure directly in JS.
 * Maps the standard pb and variable files without relying on TFJS loaders.
 *
 * @param files Record of file paths to their raw binary content.
 * @returns A structured mapping to the SavedModel binaries.
 * @throws Error if `saved_model.pb` is missing.
 */
export function parseSavedModel(files: Record<string, Uint8Array>): SavedModel {
  let savedModelPb: Uint8Array | undefined;
  let variablesIndex: Uint8Array | undefined;
  const variablesData: Uint8Array[] = [];

  for (const entry of Object.entries(files)) {
    const filename = entry[0];
    const data = entry[1];
    if (filename.endsWith('saved_model.pb')) {
      savedModelPb = data;
    } else if (filename.endsWith('variables.index')) {
      variablesIndex = data;
    } else if (filename.includes('variables.data')) {
      variablesData.push(data);
    }
  }

  if (savedModelPb === undefined) {
    throw new ConversionError('Invalid SavedModel format: missing saved_model.pb');
  }

  const result: SavedModel = {
    savedModelPb,
    variablesData,
  };

  if (variablesIndex !== undefined) {
    result.variablesIndex = variablesIndex;
  }

  return result;
}
