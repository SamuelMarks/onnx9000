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
    const typeWriter = new Writer(); /* v8 ignore next */ /* v8 ignore next */
    if (feat.type.int64Type) {
      /* v8 ignore next */ /* v8 ignore next */
      typeWriter.writeTag(1, WIRE_TYPE_LENGTH_DELIMITED); /* v8 ignore next */ /* v8 ignore next */
      typeWriter.writeVarInt(0); /* v8 ignore next */ /* v8 ignore next */
    } else if (feat.type.doubleType) {
      /* v8 ignore next */ /* v8 ignore next */
      typeWriter.writeTag(2, WIRE_TYPE_LENGTH_DELIMITED); /* v8 ignore next */ /* v8 ignore next */
      typeWriter.writeVarInt(0); /* v8 ignore next */ /* v8 ignore next */
    } else if (feat.type.stringType) {
      /* v8 ignore next */ /* v8 ignore next */
      typeWriter.writeTag(3, WIRE_TYPE_LENGTH_DELIMITED); /* v8 ignore next */ /* v8 ignore next */
      typeWriter.writeVarInt(0); /* v8 ignore next */ /* v8 ignore next */
    } else if (feat.type.imageType) {
      /* v8 ignore next */ /* v8 ignore next */
      const imgWriter = new Writer(); /* v8 ignore next */ /* v8 ignore next */
      imgWriter.writeTag(1, WIRE_TYPE_VARINT); /* v8 ignore next */ /* v8 ignore next */
      imgWriter.writeVarInt(feat.type.imageType.width); /* v8 ignore next */ /* v8 ignore next */
      imgWriter.writeTag(2, WIRE_TYPE_VARINT); /* v8 ignore next */ /* v8 ignore next */
      imgWriter.writeVarInt(feat.type.imageType.height); /* v8 ignore next */ /* v8 ignore next */
      imgWriter.writeTag(3, WIRE_TYPE_VARINT); /* v8 ignore next */ /* v8 ignore next */
      imgWriter.writeVarInt(
        feat.type.imageType.colorSpace,
      ); /* v8 ignore next */ /* v8 ignore next */
      const imgBytes = imgWriter.finish(); /* v8 ignore next */ /* v8 ignore next */
      typeWriter.writeTag(4, WIRE_TYPE_LENGTH_DELIMITED); /* v8 ignore next */ /* v8 ignore next */
      typeWriter.writeVarInt(imgBytes.length); /* v8 ignore next */ /* v8 ignore next */
      typeWriter.writeBytes(imgBytes); /* v8 ignore next */ /* v8 ignore next */
    } else if (feat.type.multiArrayType) {
      /* v8 ignore next */ /* v8 ignore next */
      const arrWriter = new Writer(); /* v8 ignore next */ /* v8 ignore next */
      for (const dim of feat.type.multiArrayType.shape) {
        /* v8 ignore next */ /* v8 ignore next */
        arrWriter.writeTag(1, WIRE_TYPE_VARINT); /* v8 ignore next */ /* v8 ignore next */
        arrWriter.writeVarInt(dim); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      arrWriter.writeTag(2, WIRE_TYPE_VARINT); /* v8 ignore next */ /* v8 ignore next */
      arrWriter.writeVarInt(
        feat.type.multiArrayType.dataType,
      ); /* v8 ignore next */ /* v8 ignore next */
      const arrBytes = arrWriter.finish(); /* v8 ignore next */ /* v8 ignore next */
      typeWriter.writeTag(5, WIRE_TYPE_LENGTH_DELIMITED); /* v8 ignore next */ /* v8 ignore next */
      typeWriter.writeVarInt(arrBytes.length); /* v8 ignore next */ /* v8 ignore next */
      typeWriter.writeBytes(arrBytes); /* v8 ignore next */ /* v8 ignore next */
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
  } /* v8 ignore next */ /* v8 ignore next */
  if (meta.creatorDefined) {
    /* v8 ignore next */ /* v8 ignore next */
    for (const key of Object.keys(meta.creatorDefined)) {
      /* v8 ignore next */ /* v8 ignore next */
      const val = meta.creatorDefined[key]; /* v8 ignore next */ /* v8 ignore next */
      if (val !== undefined) {
        /* v8 ignore next */ /* v8 ignore next */
        const kvWriter = new Writer(); /* v8 ignore next */ /* v8 ignore next */
        kvWriter.writeTag(1, WIRE_TYPE_LENGTH_DELIMITED); /* v8 ignore next */ /* v8 ignore next */
        kvWriter.writeString(key); /* v8 ignore next */ /* v8 ignore next */
        kvWriter.writeTag(2, WIRE_TYPE_LENGTH_DELIMITED); /* v8 ignore next */ /* v8 ignore next */
        kvWriter.writeString(val); /* v8 ignore next */ /* v8 ignore next */
        const kvBytes = kvWriter.finish(); /* v8 ignore next */ /* v8 ignore next */
        writer.writeTag(5, WIRE_TYPE_LENGTH_DELIMITED); /* v8 ignore next */ /* v8 ignore next */
        writer.writeVarInt(kvBytes.length); /* v8 ignore next */ /* v8 ignore next */
        writer.writeBytes(kvBytes); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  }
  return writer.finish();
}

function emitNeuralNetwork(nn: NeuralNetwork): Uint8Array {
  const writer = new Writer();
  if (nn.layers) {
    /* v8 ignore next */ /* v8 ignore next */
    for (const layer of nn.layers) {
      /* v8 ignore next */ /* v8 ignore next */
      const layerWriter = new Writer(); /* v8 ignore next */ /* v8 ignore next */
      layerWriter.writeTag(1, WIRE_TYPE_LENGTH_DELIMITED); /* v8 ignore next */ /* v8 ignore next */
      layerWriter.writeString(layer.name); /* v8 ignore next */ /* v8 ignore next */
      for (const inp of layer.input) {
        /* v8 ignore next */ /* v8 ignore next */
        layerWriter.writeTag(
          2,
          WIRE_TYPE_LENGTH_DELIMITED,
        ); /* v8 ignore next */ /* v8 ignore next */
        layerWriter.writeString(inp); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      for (const out of layer.output) {
        /* v8 ignore next */ /* v8 ignore next */
        layerWriter.writeTag(
          3,
          WIRE_TYPE_LENGTH_DELIMITED,
        ); /* v8 ignore next */ /* v8 ignore next */
        layerWriter.writeString(out); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
      const layerBytes = layerWriter.finish(); /* v8 ignore next */ /* v8 ignore next */
      writer.writeTag(1, WIRE_TYPE_LENGTH_DELIMITED); /* v8 ignore next */ /* v8 ignore next */
      writer.writeVarInt(layerBytes.length); /* v8 ignore next */ /* v8 ignore next */
      writer.writeBytes(layerBytes); /* v8 ignore next */ /* v8 ignore next */
    }
  }
  return writer.finish();
}

function emitMILSpecProgram(prog: MILSpecProgram): Uint8Array {
  const writer = new Writer();
  writer.writeTag(1, WIRE_TYPE_VARINT);
  writer.writeVarInt(prog.version);
  if (prog.functions) {
    /* v8 ignore next */ /* v8 ignore next */
    for (const funcName of Object.keys(prog.functions)) {
      /* v8 ignore next */ /* v8 ignore next */
      const funcObj = prog.functions[funcName]; /* v8 ignore next */ /* v8 ignore next */
      if (funcObj !== undefined) {
        /* v8 ignore next */ /* v8 ignore next */
        const funcWriter = new Writer(); /* v8 ignore next */ /* v8 ignore next */
        for (const inp of funcObj.inputs) {
          /* v8 ignore next */ /* v8 ignore next */
          const inpWriter = new Writer(); /* v8 ignore next */ /* v8 ignore next */
          inpWriter.writeTag(
            1,
            WIRE_TYPE_LENGTH_DELIMITED,
          ); /* v8 ignore next */ /* v8 ignore next */
          inpWriter.writeString(inp.name); /* v8 ignore next */ /* v8 ignore next */
          const inpBytes = inpWriter.finish(); /* v8 ignore next */ /* v8 ignore next */
          funcWriter.writeTag(
            1,
            WIRE_TYPE_LENGTH_DELIMITED,
          ); /* v8 ignore next */ /* v8 ignore next */
          funcWriter.writeVarInt(inpBytes.length); /* v8 ignore next */ /* v8 ignore next */
          funcWriter.writeBytes(inpBytes); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
        const funcBytes = funcWriter.finish(); /* v8 ignore next */ /* v8 ignore next */
        const entryWriter = new Writer(); /* v8 ignore next */ /* v8 ignore next */
        entryWriter.writeTag(
          1,
          WIRE_TYPE_LENGTH_DELIMITED,
        ); /* v8 ignore next */ /* v8 ignore next */
        entryWriter.writeString(funcName); /* v8 ignore next */ /* v8 ignore next */
        entryWriter.writeTag(
          2,
          WIRE_TYPE_LENGTH_DELIMITED,
        ); /* v8 ignore next */ /* v8 ignore next */
        entryWriter.writeVarInt(funcBytes.length); /* v8 ignore next */ /* v8 ignore next */
        entryWriter.writeBytes(funcBytes); /* v8 ignore next */ /* v8 ignore next */
        const entryBytes = entryWriter.finish(); /* v8 ignore next */ /* v8 ignore next */
        writer.writeTag(2, WIRE_TYPE_LENGTH_DELIMITED); /* v8 ignore next */ /* v8 ignore next */
        writer.writeVarInt(entryBytes.length); /* v8 ignore next */ /* v8 ignore next */
        writer.writeBytes(entryBytes); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    }
  }
  return writer.finish();
}
