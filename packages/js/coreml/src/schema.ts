/* eslint-disable */
import {
  Reader,
  WIRE_TYPE_VARINT,
  WIRE_TYPE_64BIT,
  WIRE_TYPE_LENGTH_DELIMITED,
  WIRE_TYPE_32BIT,
  skipField,
  readVarInt,
  readString,
  readTag,
} from '@onnx9000/core';
import { Writer } from './protobuf.js';

export interface Model {
  specificationVersion: number;
  description?: ModelDescription;
  neuralNetwork?: NeuralNetwork;
  mlProgram?: MILSpecProgram;
}

export interface ModelDescription {
  input: FeatureDescription[];
  output: FeatureDescription[];
  metadata?: Metadata;
}

export interface FeatureDescription {
  name: string;
  shortDescription?: string;
  type?: FeatureType;
}

export interface FeatureType {
  int64Type?: Int64FeatureType;
  doubleType?: DoubleFeatureType;
  stringType?: StringFeatureType;
  imageType?: ImageFeatureType;
  multiArrayType?: ArrayFeatureType;
  dictionaryType?: DictionaryFeatureType;
  sequenceType?: SequenceFeatureType;
}

export interface Int64FeatureType {}
export interface DoubleFeatureType {}
export interface StringFeatureType {}
export interface ImageFeatureType {
  width: number;
  height: number;
  colorSpace: number; // 0=Invalid, 10=Grayscale, 20=RGB, 30=BGR
}
export interface ArrayFeatureType {
  shape: number[];
  dataType: number; // 0=INVALID, 1=FLOAT32, 2=DOUBLE, 3=INT32, 4=FLOAT16
}
export interface DictionaryFeatureType {}
export interface SequenceFeatureType {}

export interface Metadata {
  shortDescription?: string;
  versionString?: string;
  author?: string;
  license?: string;
  creatorDefined: Record<string, string>;
}

export interface NeuralNetwork {
  layers: NeuralNetworkLayer[];
}

export interface NeuralNetworkLayer {
  name: string;
  input: string[];
  output: string[];
}

export interface MILSpecProgram {
  version: number;
  functions: Record<string, MILSpecFunction>;
}

export interface MILSpecFunction {
  inputs: MILSpecNamedValueType[];
  blockSpecializations: Record<string, MILSpecBlock>;
}

export interface MILSpecNamedValueType {
  name: string;
  type: MILSpecType;
}

export interface MILSpecType {
  tensorType?: MILSpecTensorType;
}

export interface MILSpecTensorType {
  dataType: number; // Float32, etc.
  rank: number;
}

export interface MILSpecBlock {
  inputs: MILSpecNamedValueType[];
  operations: MILSpecOperation[];
  outputs: string[];
}

export interface MILSpecOperation {
  type: string;
  inputs: Record<string, MILSpecArgument>;
  outputs: MILSpecNamedValueType[];
  blocks: MILSpecBlock[];
}

export interface MILSpecArgument {
  arguments: MILSpecArgumentBinding[];
}

export interface MILSpecArgumentBinding {
  name: string;
}

// Model.proto parsers / emitters

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

async function parseNeuralNetwork(reader: Reader, limit: number): Promise<NeuralNetwork> {
  const nn: NeuralNetwork = { layers: [] };
  while (reader.getPosition() < limit) {
    const { fieldNumber, wireType } = await readTag(reader);
    await skipField(reader, wireType);
  }
  return nn;
}

async function parseMILSpecProgram(reader: Reader, limit: number): Promise<MILSpecProgram> {
  const prog: MILSpecProgram = { version: 1, functions: {} as any };
  while (reader.getPosition() < limit) {
    const { fieldNumber, wireType } = await readTag(reader);
    await skipField(reader, wireType);
  }
  return prog;
}
