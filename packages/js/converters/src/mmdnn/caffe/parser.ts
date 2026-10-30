/**
 * @fileoverview Caffe prototxt text parser and binary `.caffemodel` protobuf parser.
 */
import {
  BufferReader,
  type Reader,
  readString,
  readTag,
  readVarInt,
  readVarInt64,
  skipField,
  WIRE_TYPE_32BIT,
  WIRE_TYPE_LENGTH_DELIMITED,
  WIRE_TYPE_VARINT,
} from '@onnx9000/core';
import type {
  CaffeBlob,
  CaffeLayerParam,
  CaffeNetParameter,
  CaffePrototxtObject,
  CaffePrototxtValue,
} from '../types.js';

export type { CaffeBlob, CaffeLayerParam, CaffeNetParameter };

/**
 * Parses a Caffe `.prototxt` string into a strongly typed `CaffeNetParameter` structure.
 *
 * @param text Raw Caffe prototxt string.
 * @returns Parsed Caffe network architecture as a `CaffeNetParameter` object.
 */
export function parsePrototxt(text: string): CaffeNetParameter {
  const result: CaffeNetParameter = {
    layer: [],
    input: [],
    input_dim: [],
    input_shape: [],
  };
  const lines = text.split(String.fromCharCode(10));
  const rootObj: CaffePrototxtObject = result as CaffePrototxtObject;
  const stack: CaffePrototxtObject[] = [rootObj];
  let currentObj: CaffePrototxtObject = rootObj;

  for (const rawLine of lines) {
    const line = (rawLine.split('#')[0] || '').trim();
    if (!line) continue;

    if (line.endsWith('{')) {
      const key = line.slice(0, -1).trim();
      const newObj: CaffePrototxtObject = {};

      if (key === 'layer' || key === 'layers') {
        result.layer.push(newObj as CaffeLayerParam);
        stack.push(newObj);
        currentObj = newObj;
      } else {
        const existing = currentObj[key];
        if (existing === undefined) {
          currentObj[key] = newObj;
        } else if (Array.isArray(existing)) {
          existing.push(newObj);
        } else {
          currentObj[key] = [existing, newObj];
        }
        stack.push(newObj);
        currentObj = newObj;
      }
    } else if (line === '}') {
      stack.pop();
      currentObj = stack[stack.length - 1] ?? rootObj;
    } else {
      const colonIdx = line.indexOf(':');
      if (colonIdx !== -1) {
        const key = line.substring(0, colonIdx).trim();
        const valStr = line.substring(colonIdx + 1).trim();
        let val: CaffePrototxtValue = valStr;
        if (
          (valStr.startsWith('"') && valStr.endsWith('"')) ||
          (valStr.startsWith("'") && valStr.endsWith("'"))
        ) {
          val = valStr.substring(1, valStr.length - 1);
        } else if (!Number.isNaN(Number(valStr)) && valStr !== 'true' && valStr !== 'false') {
          val = Number(valStr);
        } else if (valStr === 'true') {
          val = true;
        } else if (valStr === 'false') {
          val = false;
        }

        if (
          key === 'input' ||
          key === 'input_dim' ||
          key === 'bottom' ||
          key === 'top' ||
          key === 'dim'
        ) {
          const existing = currentObj[key];
          if (existing === undefined) {
            currentObj[key] = [val];
          } else if (Array.isArray(existing)) {
            existing.push(val);
          } else {
            currentObj[key] = [existing, val];
          }
        } else {
          currentObj[key] = val;
        }
      }
    }
  }
  return result;
}

/**
 * Parses a Caffe binary model (`.caffemodel`) from a `Uint8Array` into a `CaffeNetParameter`.
 *
 * @param buffer Binary buffer containing the Caffe model.
 * @returns Promise resolving to the parsed `CaffeNetParameter`.
 */
export async function parseCaffemodel(buffer: Uint8Array): Promise<CaffeNetParameter> {
  const reader = new BufferReader(buffer);
  const result: CaffeNetParameter = { layer: [] };

  while (reader.getPosition() < reader.getLength()) {
    const { fieldNumber, wireType } = await readTag(reader);

    if (fieldNumber === 1 && wireType === WIRE_TYPE_LENGTH_DELIMITED) {
      const len = await readVarInt(reader);
      result.name = await readString(reader, len);
    } else if (
      (fieldNumber === 100 || fieldNumber === 2) &&
      wireType === WIRE_TYPE_LENGTH_DELIMITED
    ) {
      const len = await readVarInt(reader);
      const subReader = new BufferReader(await reader.readBytes(len));
      const layer = await parseLayerParameter(subReader);
      result.layer.push(layer);
    } else {
      await skipField(reader, wireType);
    }
  }

  return result;
}

/**
 * Parses a single `LayerParameter` from a protobuf reader.
 *
 * @param reader Protobuf reader positioned at a `LayerParameter`.
 * @returns Promise resolving to the parsed `CaffeLayerParam`.
 */
async function parseLayerParameter(reader: Reader): Promise<CaffeLayerParam> {
  const blobs: CaffeBlob[] = [];
  const layer: CaffeLayerParam = { blobs };

  while (reader.getPosition() < reader.getLength()) {
    const { fieldNumber, wireType } = await readTag(reader);

    if (fieldNumber === 1 && wireType === WIRE_TYPE_LENGTH_DELIMITED) {
      const len = await readVarInt(reader);
      layer.name = await readString(reader, len);
    } else if (fieldNumber === 2 && wireType === WIRE_TYPE_LENGTH_DELIMITED) {
      const len = await readVarInt(reader);
      layer.type = await readString(reader, len);
    } else if (
      (fieldNumber === 50 || fieldNumber === 6) &&
      wireType === WIRE_TYPE_LENGTH_DELIMITED
    ) {
      const len = await readVarInt(reader);
      const subReader = new BufferReader(await reader.readBytes(len));
      const blob = await parseBlobProto(subReader);
      blobs.push(blob);
    } else {
      await skipField(reader, wireType);
    }
  }
  return layer;
}

/**
 * Parses a `BlobProto` from a protobuf reader.
 *
 * @param reader Protobuf reader positioned at a `BlobProto`.
 * @returns Promise resolving to the parsed `CaffeBlob` containing shape and data.
 */
async function parseBlobProto(reader: Reader): Promise<CaffeBlob> {
  const blob: CaffeBlob = { data: [] };
  const shape: number[] = [];

  while (reader.getPosition() < reader.getLength()) {
    const { fieldNumber, wireType } = await readTag(reader);

    if (fieldNumber === 1 && wireType === WIRE_TYPE_VARINT) {
      shape[0] = await readVarInt(reader);
    } else if (fieldNumber === 2 && wireType === WIRE_TYPE_VARINT) {
      shape[1] = await readVarInt(reader);
    } else if (fieldNumber === 3 && wireType === WIRE_TYPE_VARINT) {
      shape[2] = await readVarInt(reader);
    } else if (fieldNumber === 4 && wireType === WIRE_TYPE_VARINT) {
      shape[3] = await readVarInt(reader);
    } else if (fieldNumber === 5) {
      if (wireType === WIRE_TYPE_32BIT) {
        const bytes = await reader.readBytes(4);
        const floatVal = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getFloat32(
          0,
          true,
        );
        blob.data.push(floatVal);
      } else if (wireType === WIRE_TYPE_LENGTH_DELIMITED) {
        const len = await readVarInt(reader);
        const bytes = await reader.readBytes(len);
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        for (let i = 0; i < len; i += 4) {
          blob.data.push(view.getFloat32(i, true));
        }
      } else {
        await skipField(reader, wireType);
      }
    } else if (fieldNumber === 7 && wireType === WIRE_TYPE_LENGTH_DELIMITED) {
      const len = await readVarInt(reader);
      const subReader = new BufferReader(await reader.readBytes(len));
      blob.shape = await parseBlobShape(subReader);
    } else {
      await skipField(reader, wireType);
    }
  }

  if (!blob.shape && shape.length > 0) {
    blob.shape = shape.filter((d) => d !== undefined);
  }
  return blob;
}

/**
 * Parses a `BlobShape` from a protobuf reader.
 *
 * @param reader Protobuf reader positioned at a `BlobShape`.
 * @returns Promise resolving to an array of dimension sizes.
 */
async function parseBlobShape(reader: Reader): Promise<number[]> {
  const shape: number[] = [];
  while (reader.getPosition() < reader.getLength()) {
    const { fieldNumber, wireType } = await readTag(reader);
    if (fieldNumber === 1 && wireType === WIRE_TYPE_VARINT) {
      shape.push(Number(await readVarInt64(reader)));
    } else if (fieldNumber === 1 && wireType === WIRE_TYPE_LENGTH_DELIMITED) {
      const len = await readVarInt(reader);
      const endPos = reader.getPosition() + len;
      while (reader.getPosition() < endPos) {
        shape.push(Number(await readVarInt64(reader)));
      }
    } else {
      await skipField(reader, wireType);
    }
  }
  return shape;
}
