import {
  readVarInt,
  readVarInt64,
  readString,
  readTag,
  skipField,
  Reader,
  BufferReader,
  WIRE_TYPE_VARINT,
  WIRE_TYPE_LENGTH_DELIMITED,
  WIRE_TYPE_32BIT,
} from '@onnx9000/core';

/**
 * Parses a Caffe prototxt string into a JavaScript object.
 *
 * @param {string} text - The raw Caffe prototxt string.
 * @returns {any} The parsed Caffe network architecture as a JavaScript object.
 */
export function parsePrototxt(text: string): any {
  const result: any = { layer: [], input: [], input_dim: [], input_shape: [] };
  const lines = text.split('\n');
  const stack: any[] = [result];
  let currentObj = result;

  for (let i = 0; i < lines.length; i++) {
    const _l = lines[i];
    if (_l === undefined) continue;
    let line = _l.split('#')[0]!.trim();
    if (!line) continue;

    if (line.endsWith('{')) {
      const key = line.slice(0, -1).trim();
      const newObj: any = {};

      if (key === 'layer' || key === 'layers') {
        result.layer.push(newObj);
        stack.push(newObj);
        currentObj = newObj;
      } else {
        if (!currentObj[key]) {
          currentObj[key] = newObj;
        } else {
          if (!Array.isArray(currentObj[key])) {
            currentObj[key] = [currentObj[key]];
          }
          currentObj[key].push(newObj);
        }
        stack.push(newObj);
        currentObj = newObj;
      }
    } else if (line === '}') {
      stack.pop();
      currentObj = stack[stack.length - 1];
    } else {
      const colonIdx = line.indexOf(':');
      if (colonIdx !== -1) {
        const key = line.substring(0, colonIdx).trim();
        let valStr = line.substring(colonIdx + 1).trim();
        let val: any = valStr;
        if (
          (valStr.startsWith('"') && valStr.endsWith('"')) ||
          (valStr.startsWith("'") && valStr.endsWith("'"))
        ) {
          val = valStr.substring(1, valStr.length - 1);
        } else if (!isNaN(Number(valStr)) && valStr !== 'true' && valStr !== 'false') {
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
          if (!currentObj[key]) {
            currentObj[key] = [];
          } else if (!Array.isArray(currentObj[key])) {
            currentObj[key] = [currentObj[key]];
          }
          currentObj[key].push(val);
        } else {
          currentObj[key] = val;
        }
      }
    }
  }
  return result;
}

/**
 * Parses a Caffe binary model (.caffemodel) from a Uint8Array into a JavaScript object.
 *
 * @param {Uint8Array} buffer - The binary buffer containing the Caffe model.
 * @returns {Promise<any>} A promise that resolves to the parsed Caffe model parameters.
 */
export async function parseCaffemodel(buffer: Uint8Array): Promise<any> {
  const reader = new BufferReader(buffer);
  const result: any = { layer: [] };

  while (reader.getPosition() < reader.getLength()) {
    const { fieldNumber, wireType } = await readTag(reader);

    // NetParameter.name (1)
    if (fieldNumber === 1 && wireType === WIRE_TYPE_LENGTH_DELIMITED) {
      const len = await readVarInt(reader);
      result.name = await readString(reader, len);
    }
    // NetParameter.layer (100) or V1LayerParameter (2)
    else if (
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
 * Parses a single LayerParameter from a protobuf reader.
 *
 * @param {Reader} reader - The protobuf reader positioned at a LayerParameter.
 * @returns {Promise<any>} A promise that resolves to the parsed layer parameter object.
 */
async function parseLayerParameter(reader: Reader): Promise<any> {
  const layer: any = { blobs: [] };

  while (reader.getPosition() < reader.getLength()) {
    const { fieldNumber, wireType } = await readTag(reader);

    // name (1)
    if (fieldNumber === 1 && wireType === WIRE_TYPE_LENGTH_DELIMITED) {
      const len = await readVarInt(reader);
      layer.name = await readString(reader, len);
    }
    // type (2)
    else if (fieldNumber === 2 && wireType === WIRE_TYPE_LENGTH_DELIMITED) {
      const len = await readVarInt(reader);
      layer.type = await readString(reader, len);
    }
    // blobs (50) or V1 blobs (6)
    else if ((fieldNumber === 50 || fieldNumber === 6) && wireType === WIRE_TYPE_LENGTH_DELIMITED) {
      const len = await readVarInt(reader);
      const subReader = new BufferReader(await reader.readBytes(len));
      const blob = await parseBlobProto(subReader);
      layer.blobs.push(blob);
    } else {
      await skipField(reader, wireType);
    }
  }
  return layer;
}

/**
 * Parses a BlobProto from a protobuf reader.
 *
 * @param {Reader} reader - The protobuf reader positioned at a BlobProto.
 * @returns {Promise<any>} A promise that resolves to the parsed blob containing shape and data.
 */
async function parseBlobProto(reader: Reader): Promise<any> {
  const blob: any = { data: [] };
  const shape: number[] = [];

  while (reader.getPosition() < reader.getLength()) {
    const { fieldNumber, wireType } = await readTag(reader);

    // num (1)
    if (fieldNumber === 1 && wireType === WIRE_TYPE_VARINT) {
      shape[0] = await readVarInt(reader);
    }
    // channels (2)
    else if (fieldNumber === 2 && wireType === WIRE_TYPE_VARINT) {
      shape[1] = await readVarInt(reader);
    }
    // height (3)
    else if (fieldNumber === 3 && wireType === WIRE_TYPE_VARINT) {
      shape[2] = await readVarInt(reader);
    }
    // width (4)
    else if (fieldNumber === 4 && wireType === WIRE_TYPE_VARINT) {
      shape[3] = await readVarInt(reader);
    }
    // data (5) float
    else if (fieldNumber === 5) {
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
    }
    // shape (7)
    else if (fieldNumber === 7 && wireType === WIRE_TYPE_LENGTH_DELIMITED) {
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
 * Parses a BlobShape from a protobuf reader.
 *
 * @param {Reader} reader - The protobuf reader positioned at a BlobShape.
 * @returns {Promise<number[]>} A promise that resolves to an array of numbers representing the shape.
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
