/* eslint-disable */
// @ts-nocheck
/**
 * Defines the supported mathematical data types natively supported by Web-Native machine learning layers.
 */
export type DataType = 'float32' | 'int32' | 'bool' | 'complex64' | 'string' | 'float16' | 'uint8';

/**
 * Represents a single weight entry in a TF.js manifest.
 */
export interface WeightManifestEntry {
  name: string;
  shape: number[];
  dtype: DataType;
  quantization?: {
    scale: number;
    min: number;
    dtype: string;
  };
}

/**
 * Represents a group of weight files (paths) and their associated manifests.
 */
export interface WeightGroup {
  paths: string[];
  weights: WeightManifestEntry[];
}

/**
 * Union type for JSON primitive values.
 */
export type JsonValue = string | number | boolean | null | JsonArray | JsonObject;
/**
 * Array type for JSON values.
 */
export type JsonArray = JsonValue[];
/**
 * Object type for JSON structures.
 */
export interface JsonObject {
  [key: string]: JsonValue;
}

/**
 * Represents a TensorFlow.js 'layers-model' (analogous to Keras H5 or SavedModel topologies).
 */
export interface TFJSLayersModel {
  format: 'layers-model';
  generatedBy?: string;
  convertedBy?: string;
  modelTopology: JsonObject; // The raw Keras JSON config
  weightsManifest: WeightGroup[];
}

/**
 * Represents a TensorFlow.js 'graph-model' (lower level frozen execution graphs).
 */
export interface TFJSGraphModel {
  format: 'graph-model';
  generatedBy?: string;
  convertedBy?: string;
  modelTopology: {
    node: JsonObject[];
    versions?: JsonObject;
  };
  weightsManifest: WeightGroup[];
}

/**
 * A combined union representing any supported TF.js model format.
 */
export type TFJSModel = TFJSLayersModel | TFJSGraphModel;

/**
 * Parses a raw JSON string into a structured TFJSModel.
 * Automatically attempts to infer format if it's missing from the top level.
 *
 * @param jsonText The raw JSON model file content.
 * @returns The strongly-typed TFJSModel representation.
 * @throws Error if the model format is unsupported or unrecognized.
 */
export function parseTFJSModel(jsonText: string): TFJSModel {
  const parsed = JSON.parse(jsonText) as JsonObject;
  if (parsed['format'] !== 'layers-model' && parsed['format'] !== 'graph-model') {
    const top = parsed['modelTopology'] as JsonObject | undefined;
    if (top !== undefined && top['node'] !== undefined) {
      parsed['format'] = 'graph-model';
    } else if (top !== undefined && top['class_name'] !== undefined) {
      parsed['format'] = 'layers-model';
    } else {
      throw new Error('Unsupported or unrecognized TF.js model format');
    }
  }
  const manifestArray = parsed['weightsManifest'] as JsonArray;
  const weightsManifest: WeightGroup[] = manifestArray.map((groupVal) => {
    const groupObj = groupVal as JsonObject;
    const pathsArr = groupObj['paths'] as JsonArray;
    const weightsArr = groupObj['weights'] as JsonArray;
    return {
      paths: pathsArr.map((p) => p as string),
      weights: weightsArr.map((wVal) => {
        const wObj = wVal as JsonObject;
        const shapeArr = wObj['shape'] as JsonArray;
        const entry: WeightManifestEntry = {
          name: wObj['name'] as string,
          shape: shapeArr.map((s) => s as number),
          dtype: wObj['dtype'] as DataType,
        };
        if (wObj['quantization']) {
          entry.quantization = wObj['quantization'] as JsonObject as {
            scale: number;
            min: number;
            dtype: string;
          };
        }
        return entry;
      }),
    };
  });

  return {
    format: parsed['format'] as 'layers-model' | 'graph-model',
    generatedBy: typeof parsed['generatedBy'] === 'string' ? parsed['generatedBy'] : undefined,
    convertedBy: typeof parsed['convertedBy'] === 'string' ? parsed['convertedBy'] : undefined,
    modelTopology: parsed['modelTopology'] as JsonObject,
    weightsManifest,
  } as TFJSModel;
}
