import { Writer } from './protobuf.js';
import {
  Model,
  ModelDescription,
  FeatureDescription,
  Metadata,
  NeuralNetwork,
  MILSpecProgram,
  MILSpecFunction,
  MILSpecBlock,
} from './schema.js';
import { WIRE_TYPE_VARINT, WIRE_TYPE_LENGTH_DELIMITED } from '@onnx9000/core';

export function emitModel(model: Model): Uint8Array {
  const writer = new Writer();
  if (model.specificationVersion !== undefined) {
    writer.writeTag(1, WIRE_TYPE_VARINT);
    writer.writeVarInt(model.specificationVersion);
  }
  if (model.description) {
    const descBytes = emitModelDescription(model.description);
    writer.writeTag(2, WIRE_TYPE_LENGTH_DELIMITED);
    writer.writeVarInt(descBytes.length);
    writer.writeBytes(descBytes);
  }
  if (model.neuralNetwork) {
    const nnBytes = emitNeuralNetwork(model.neuralNetwork);
    writer.writeTag(6, WIRE_TYPE_LENGTH_DELIMITED);
    writer.writeVarInt(nnBytes.length);
    writer.writeBytes(nnBytes);
  } else if (model.mlProgram) {
    const mlBytes = emitMILSpecProgram(model.mlProgram);
    writer.writeTag(68, WIRE_TYPE_LENGTH_DELIMITED);
    writer.writeVarInt(mlBytes.length);
    writer.writeBytes(mlBytes);
  }
  return writer.finish();
}

function emitModelDescription(desc: ModelDescription): Uint8Array {
  const writer = new Writer();
  for (const input of desc.input) {
    const inputBytes = emitFeatureDescription(input);
    writer.writeTag(1, WIRE_TYPE_LENGTH_DELIMITED);
    writer.writeVarInt(inputBytes.length);
    writer.writeBytes(inputBytes);
  }
  for (const output of desc.output) {
    const outputBytes = emitFeatureDescription(output);
    writer.writeTag(10, WIRE_TYPE_LENGTH_DELIMITED);
    writer.writeVarInt(outputBytes.length);
    writer.writeBytes(outputBytes);
  }
  if (desc.metadata) {
    const metaBytes = emitMetadata(desc.metadata);
    writer.writeTag(100, WIRE_TYPE_LENGTH_DELIMITED);
    writer.writeVarInt(metaBytes.length);
    writer.writeBytes(metaBytes);
  }
  return writer.finish();
}

function emitFeatureDescription(feat: FeatureDescription): Uint8Array {
  const writer = new Writer();
  if (feat.name) {
    writer.writeTag(1, WIRE_TYPE_LENGTH_DELIMITED);
    writer.writeString(feat.name);
  }
  if (feat.shortDescription) {
    writer.writeTag(2, WIRE_TYPE_LENGTH_DELIMITED);
    writer.writeString(feat.shortDescription);
  }
  // TODO: emit feature types
  return writer.finish();
}

function emitMetadata(meta: Metadata): Uint8Array {
  const writer = new Writer();
  if (meta.shortDescription) {
    writer.writeTag(1, WIRE_TYPE_LENGTH_DELIMITED);
    writer.writeString(meta.shortDescription);
  }
  if (meta.versionString) {
    writer.writeTag(2, WIRE_TYPE_LENGTH_DELIMITED);
    writer.writeString(meta.versionString);
  }
  if (meta.author) {
    writer.writeTag(3, WIRE_TYPE_LENGTH_DELIMITED);
    writer.writeString(meta.author);
  }
  if (meta.license) {
    writer.writeTag(4, WIRE_TYPE_LENGTH_DELIMITED);
    writer.writeString(meta.license);
  }
  // TODO: user defined metadata mapping
  return writer.finish();
}

function emitNeuralNetwork(nn: NeuralNetwork): Uint8Array {
  const writer = new Writer();
  // TODO: layers
  return writer.finish();
}

function emitMILSpecProgram(prog: MILSpecProgram): Uint8Array {
  const writer = new Writer();
  writer.writeTag(1, WIRE_TYPE_VARINT);
  writer.writeVarInt(prog.version);
  // TODO: functions map
  return writer.finish();
}
