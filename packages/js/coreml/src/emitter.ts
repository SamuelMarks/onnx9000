/* eslint-disable */
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
  if (feat.type) {
    const typeWriter = new Writer();
    if (feat.type.int64Type) {
      typeWriter.writeTag(1, WIRE_TYPE_LENGTH_DELIMITED);
      typeWriter.writeVarInt(0);
    } else if (feat.type.doubleType) {
      typeWriter.writeTag(2, WIRE_TYPE_LENGTH_DELIMITED);
      typeWriter.writeVarInt(0);
    } else if (feat.type.stringType) {
      typeWriter.writeTag(3, WIRE_TYPE_LENGTH_DELIMITED);
      typeWriter.writeVarInt(0);
    } else if (feat.type.imageType) {
      const imgWriter = new Writer();
      imgWriter.writeTag(1, WIRE_TYPE_VARINT);
      imgWriter.writeVarInt(feat.type.imageType.width);
      imgWriter.writeTag(2, WIRE_TYPE_VARINT);
      imgWriter.writeVarInt(feat.type.imageType.height);
      imgWriter.writeTag(3, WIRE_TYPE_VARINT);
      imgWriter.writeVarInt(feat.type.imageType.colorSpace);
      const imgBytes = imgWriter.finish();
      typeWriter.writeTag(4, WIRE_TYPE_LENGTH_DELIMITED);
      typeWriter.writeVarInt(imgBytes.length);
      typeWriter.writeBytes(imgBytes);
    } else if (feat.type.multiArrayType) {
      const arrWriter = new Writer();
      for (const dim of feat.type.multiArrayType.shape) {
        arrWriter.writeTag(1, WIRE_TYPE_VARINT);
        arrWriter.writeVarInt(dim);
      }
      arrWriter.writeTag(2, WIRE_TYPE_VARINT);
      arrWriter.writeVarInt(feat.type.multiArrayType.dataType);
      const arrBytes = arrWriter.finish();
      typeWriter.writeTag(5, WIRE_TYPE_LENGTH_DELIMITED);
      typeWriter.writeVarInt(arrBytes.length);
      typeWriter.writeBytes(arrBytes);
    }
    const typeBytes = typeWriter.finish();
    writer.writeTag(3, WIRE_TYPE_LENGTH_DELIMITED);
    writer.writeVarInt(typeBytes.length);
    writer.writeBytes(typeBytes);
  }
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
  if (meta.creatorDefined) {
    for (const key of Object.keys(meta.creatorDefined)) {
      const val = meta.creatorDefined[key];
      if (val !== undefined) {
        const kvWriter = new Writer();
        kvWriter.writeTag(1, WIRE_TYPE_LENGTH_DELIMITED);
        kvWriter.writeString(key);
        kvWriter.writeTag(2, WIRE_TYPE_LENGTH_DELIMITED);
        kvWriter.writeString(val);
        const kvBytes = kvWriter.finish();
        writer.writeTag(5, WIRE_TYPE_LENGTH_DELIMITED);
        writer.writeVarInt(kvBytes.length);
        writer.writeBytes(kvBytes);
      }
    }
  }
  return writer.finish();
}

function emitNeuralNetwork(nn: NeuralNetwork): Uint8Array {
  const writer = new Writer();
  if (nn.layers) {
    for (const layer of nn.layers) {
      const layerWriter = new Writer();
      layerWriter.writeTag(1, WIRE_TYPE_LENGTH_DELIMITED);
      layerWriter.writeString(layer.name);
      for (const inp of layer.input) {
        layerWriter.writeTag(2, WIRE_TYPE_LENGTH_DELIMITED);
        layerWriter.writeString(inp);
      }
      for (const out of layer.output) {
        layerWriter.writeTag(3, WIRE_TYPE_LENGTH_DELIMITED);
        layerWriter.writeString(out);
      }
      const layerBytes = layerWriter.finish();
      writer.writeTag(1, WIRE_TYPE_LENGTH_DELIMITED);
      writer.writeVarInt(layerBytes.length);
      writer.writeBytes(layerBytes);
    }
  }
  return writer.finish();
}

function emitMILSpecProgram(prog: MILSpecProgram): Uint8Array {
  const writer = new Writer();
  writer.writeTag(1, WIRE_TYPE_VARINT);
  writer.writeVarInt(prog.version);
  if (prog.functions) {
    for (const funcName of Object.keys(prog.functions)) {
      const funcObj = prog.functions[funcName];
      if (funcObj !== undefined) {
        const funcWriter = new Writer();
        for (const inp of funcObj.inputs) {
          const inpWriter = new Writer();
          inpWriter.writeTag(1, WIRE_TYPE_LENGTH_DELIMITED);
          inpWriter.writeString(inp.name);
          const inpBytes = inpWriter.finish();
          funcWriter.writeTag(1, WIRE_TYPE_LENGTH_DELIMITED);
          funcWriter.writeVarInt(inpBytes.length);
          funcWriter.writeBytes(inpBytes);
        }
        const funcBytes = funcWriter.finish();
        const entryWriter = new Writer();
        entryWriter.writeTag(1, WIRE_TYPE_LENGTH_DELIMITED);
        entryWriter.writeString(funcName);
        entryWriter.writeTag(2, WIRE_TYPE_LENGTH_DELIMITED);
        entryWriter.writeVarInt(funcBytes.length);
        entryWriter.writeBytes(funcBytes);
        const entryBytes = entryWriter.finish();
        writer.writeTag(2, WIRE_TYPE_LENGTH_DELIMITED);
        writer.writeVarInt(entryBytes.length);
        writer.writeBytes(entryBytes);
      }
    }
  }
  return writer.finish();
}
