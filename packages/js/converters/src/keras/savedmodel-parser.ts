/**
 * Extracted raw representations from a TF SavedModel directory
 */
export interface SavedModel {
  savedModelPb: Uint8Array;
  variablesIndex?: Uint8Array;
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
  let savedModelPb: Uint8Array | undefined = undefined;
  let variablesIndex: Uint8Array | undefined = undefined;
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
    throw new Error('Invalid SavedModel format: missing saved_model.pb');
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
