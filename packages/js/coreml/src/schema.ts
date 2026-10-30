/**
 * @fileoverview schema.ts
 * Provides schema functionality for the coreml package.
 */
import { type Reader, readString, readTag, readVarInt, skipField } from '@onnx9000/core';

/** Interface `Model` definition. */
export interface Model {
  /** Property `specificationVersion`. */
  specificationVersion: number;
  /** Property `description`. */
  description?: ModelDescription;
  /** Property `neuralNetwork`. */
  neuralNetwork?: NeuralNetwork;
  /** Property `mlProgram`. */
  mlProgram?: MILSpecProgram;
}

/** Interface `ModelDescription` definition. */
export interface ModelDescription {
  /** Property `input`. */
  input: FeatureDescription[];
  /** Property `output`. */
  output: FeatureDescription[];
  /** Property `metadata`. */
  metadata?: Metadata;
}

/** Interface `FeatureDescription` definition. */
export interface FeatureDescription {
  /** Property `name`. */
  name: string;
  /** Property `shortDescription`. */
  shortDescription?: string;
  /** Property `type`. */
  type?: FeatureType;
}

/** Interface `FeatureType` definition. */
export interface FeatureType {
  /** Property `int64Type`. */
  int64Type?: Int64FeatureType;
  /** Property `doubleType`. */
  doubleType?: DoubleFeatureType;
  /** Property `stringType`. */
  stringType?: StringFeatureType;
  /** Property `imageType`. */
  imageType?: ImageFeatureType;
  /** Property `multiArrayType`. */
  multiArrayType?: ArrayFeatureType;
  /** Property `dictionaryType`. */
  dictionaryType?: DictionaryFeatureType;
  /** Property `sequenceType`. */
  sequenceType?: SequenceFeatureType;
}

/** Type alias `Int64FeatureType`. */
export type Int64FeatureType = Record<string, never>;
/** Type alias `DoubleFeatureType`. */
export type DoubleFeatureType = Record<string, never>;
/** Type alias `StringFeatureType`. */
export type StringFeatureType = Record<string, never>;
/** Interface `ImageFeatureType` definition. */
export interface ImageFeatureType {
  /** Property `width`. */
  width: number;
  /** Property `height`. */
  height: number;
  /** Property `colorSpace`. */
  colorSpace: number; // 0=Invalid, 10=Grayscale, 20=RGB, 30=BGR
}
/** Interface `ArrayFeatureType` definition. */
export interface ArrayFeatureType {
  /** Property `shape`. */
  shape: number[];
  /** Property `dataType`. */
  dataType: number; // 0=INVALID, 1=FLOAT32, 2=DOUBLE, 3=INT32, 4=FLOAT16
}
/** Type alias `DictionaryFeatureType`. */
export type DictionaryFeatureType = Record<string, never>;
/** Type alias `SequenceFeatureType`. */
export type SequenceFeatureType = Record<string, never>;

/** Interface `Metadata` definition. */
export interface Metadata {
  /** Property `shortDescription`. */
  shortDescription?: string;
  /** Property `versionString`. */
  versionString?: string;
  /** Property `author`. */
  author?: string;
  /** Property `license`. */
  license?: string;
  /** Property `creatorDefined`. */
  creatorDefined: Record<string, string>;
}

/** Interface `NeuralNetwork` definition. */
export interface NeuralNetwork {
  /** Property `layers`. */
  layers: NeuralNetworkLayer[];
}

/** Interface `NeuralNetworkLayer` definition. */
export interface NeuralNetworkLayer {
  /** Property `name`. */
  name: string;
  /** Property `input`. */
  input: string[];
  /** Property `output`. */
  output: string[];
}

/** Interface `MILSpecProgram` definition. */
export interface MILSpecProgram {
  /** Property `version`. */
  version: number;
  /** Property `functions`. */
  functions: Record<string, MILSpecFunction>;
}

/** Interface `MILSpecFunction` definition. */
export interface MILSpecFunction {
  /** Property `inputs`. */
  inputs: MILSpecNamedValueType[];
  /** Property `blockSpecializations`. */
  blockSpecializations: Record<string, MILSpecBlock>;
}

/** Interface `MILSpecNamedValueType` definition. */
export interface MILSpecNamedValueType {
  /** Property `name`. */
  name: string;
  /** Property `type`. */
  type: MILSpecType;
}

/** Interface `MILSpecType` definition. */
export interface MILSpecType {
  /** Property `tensorType`. */
  tensorType?: MILSpecTensorType;
}

/** Interface `MILSpecTensorType` definition. */
export interface MILSpecTensorType {
  /** Property `dataType`. */
  dataType: number; // Float32, etc.
  /** Property `rank`. */
  rank: number;
}

/** Interface `MILSpecBlock` definition. */
export interface MILSpecBlock {
  /** Property `inputs`. */
  inputs: MILSpecNamedValueType[];
  /** Property `operations`. */
  operations: MILSpecOperation[];
  /** Property `outputs`. */
  outputs: string[];
}

/** Interface `MILSpecOperation` definition. */
export interface MILSpecOperation {
  /** Property `type`. */
  type: string;
  /** Property `inputs`. */
  inputs: Record<string, MILSpecArgument>;
  /** Property `outputs`. */
  outputs: MILSpecNamedValueType[];
  /** Property `blocks`. */
  blocks: MILSpecBlock[];
}

/** Interface `MILSpecArgument` definition. */
export interface MILSpecArgument {
  /** Property `arguments`. */
  arguments: MILSpecArgumentBinding[];
}

/** Interface `MILSpecArgumentBinding` definition. */
export interface MILSpecArgumentBinding {
  /** Property `name`. */
  name: string;
}

// Model.proto parsers / emitters

/**
 * Executes `parseModel`.
 * @param reader Parameter `reader`.
 * @returns Result of `parseModel`.
 */
export async function parseModel(reader: Reader): Promise<Model> {
  const model: Model = { specificationVersion: 1 };
  while (reader.getPosition() < reader.getLength()) {
    const { fieldNumber, wireType } = await readTag(reader);
    switch (fieldNumber) {
      case 1:
        model.specificationVersion = await readVarInt(reader);
        break;
      case 2: {
        const length = await readVarInt(reader);
        const end = reader.getPosition() + length;
        model.description = await parseModelDescription(reader, end);
        break;
      }
      case 6: {
        const length = await readVarInt(reader);
        const end = reader.getPosition() + length;
        model.neuralNetwork = await parseNeuralNetwork(reader, end);
        break;
      }
      case 68: {
        const length = await readVarInt(reader);
        const end = reader.getPosition() + length;
        model.mlProgram = await parseMILSpecProgram(reader, end);
        break;
      }
      default:
        await skipField(reader, wireType);
    }
  }
  return model;
}

/**
 * Executes `parseModelDescription`.
 * @param reader Parameter `reader`.
 * @param limit Parameter `limit`.
 * @returns Result of `parseModelDescription`.
 */
async function parseModelDescription(reader: Reader, limit: number): Promise<ModelDescription> {
  const desc: ModelDescription = { input: [], output: [] };
  while (reader.getPosition() < limit) {
    const { fieldNumber, wireType } = await readTag(reader);
    switch (fieldNumber) {
      case 1: {
        const length = await readVarInt(reader);
        const end = reader.getPosition() + length;
        desc.input.push(await parseFeatureDescription(reader, end));
        break;
      }
      case 10: {
        const length = await readVarInt(reader);
        const end = reader.getPosition() + length;
        desc.output.push(await parseFeatureDescription(reader, end));
        break;
      }
      case 100: {
        const length = await readVarInt(reader);
        const end = reader.getPosition() + length;
        desc.metadata = await parseMetadata(reader, end);
        break;
      }
      default:
        await skipField(reader, wireType);
    }
  }
  return desc;
}

/**
 * Executes `parseFeatureDescription`.
 * @param reader Parameter `reader`.
 * @param limit Parameter `limit`.
 * @returns Result of `parseFeatureDescription`.
 */
async function parseFeatureDescription(reader: Reader, limit: number): Promise<FeatureDescription> {
  const feat: FeatureDescription = { name: '' };
  while (reader.getPosition() < limit) {
    const { fieldNumber, wireType } = await readTag(reader);
    switch (fieldNumber) {
      case 1: {
        const length = await readVarInt(reader);
        feat.name = await readString(reader, length);
        break;
      }
      case 2: {
        const length = await readVarInt(reader);
        feat.shortDescription = await readString(reader, length);
        break;
      }
      // Type is 3... skip for now to save space, but handle properly later.
      default:
        await skipField(reader, wireType);
    }
  }
  return feat;
}

/**
 * Executes `parseMetadata`.
 * @param reader Parameter `reader`.
 * @param limit Parameter `limit`.
 * @returns Result of `parseMetadata`.
 */
async function parseMetadata(reader: Reader, limit: number): Promise<Metadata> {
  const meta: Metadata = { creatorDefined: {} };
  while (reader.getPosition() < limit) {
    const { fieldNumber, wireType } = await readTag(reader);
    switch (fieldNumber) {
      case 1: {
        const length = await readVarInt(reader);
        meta.shortDescription = await readString(reader, length);
        break;
      }
      case 2: {
        const length = await readVarInt(reader);
        meta.versionString = await readString(reader, length);
        break;
      }
      case 3: {
        const length = await readVarInt(reader);
        meta.author = await readString(reader, length);
        break;
      }
      case 4: {
        const length = await readVarInt(reader);
        meta.license = await readString(reader, length);
        break;
      }
      default:
        await skipField(reader, wireType);
    }
  }
  return meta;
}

/**
 * Executes `parseNeuralNetwork`.
 * @param reader Parameter `reader`.
 * @param limit Parameter `limit`.
 * @returns Result of `parseNeuralNetwork`.
 */
async function parseNeuralNetwork(reader: Reader, limit: number): Promise<NeuralNetwork> {
  const nn: NeuralNetwork = { layers: [] };
  while (reader.getPosition() < limit) {
    const { fieldNumber: _fieldNumber, wireType } = await readTag(reader);
    await skipField(reader, wireType);
  }
  return nn;
}

/**
 * Executes `parseMILSpecProgram`.
 * @param reader Parameter `reader`.
 * @param limit Parameter `limit`.
 * @returns Result of `parseMILSpecProgram`.
 */
async function parseMILSpecProgram(reader: Reader, limit: number): Promise<MILSpecProgram> {
  const prog: MILSpecProgram = { version: 1, functions: {} };
  while (reader.getPosition() < limit) {
    const { fieldNumber: _fieldNumber, wireType } = await readTag(reader);
    await skipField(reader, wireType);
  }
  return prog;
}
