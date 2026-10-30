/**
 * @fileoverview MXNet symbol JSON and binary parameter buffer parsers.
 */
import { ConversionError } from '@onnx9000/core';
import type { MxNetNode, MxNetParamTensor, MxNetSymbol } from '../types.js';

export type { MxNetNode, MxNetParamTensor, MxNetSymbol };

/**
 * Parses an MXNet `-symbol.json` string into a typed `MxNetSymbol` structure.
 * @param jsonStr Raw JSON string from an MXNet symbol file.
 * @returns Parsed `MxNetSymbol` graph representation.
 */
export function parseMxNetSymbol(jsonStr: string): MxNetSymbol {
  return JSON.parse(jsonStr) as MxNetSymbol;
}

/**
 * Parses an MXNet `.params` binary buffer into a dictionary of named parameter tensors.
 * @param buffer Raw binary buffer containing MXNet NDArray parameter blocks.
 * @returns Map of parameter names to parsed `MxNetParamTensor` entries.
 * @throws {Error} If magic numbers, dtype codes, or array/name counts are invalid.
 */
export function parseMxNetParams(buffer: Uint8Array): Record<string, MxNetParamTensor> {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  let offset = 0;

  const magic = view.getBigUint64(offset, true);
  offset += 8;

  if (magic !== 274n && magic !== 0x112n) {
    throw new ConversionError(`Invalid MXNet params magic: ${magic.toString(16)}`);
  }

  let count: bigint;
  const nextUint64 = view.getBigUint64(offset, true);
  if (nextUint64 === 0n) {
    offset += 8;
    count = view.getBigUint64(offset, true);
    offset += 8;
  } else {
    count = nextUint64;
    offset += 8;
  }

  const numArrays = Number(count);
  const arrays: MxNetParamTensor[] = [];

  for (let i = 0; i < numArrays; i++) {
    const tensorMagic = view.getUint32(offset, true);
    offset += 4;

    if (tensorMagic !== 0xf993fac9 && tensorMagic !== 0x0112) {
      throw new ConversionError(`Invalid MXNet NDArray block magic: ${tensorMagic.toString(16)}`);
    }

    if (tensorMagic === 0xf993fac9) {
      offset += 4;
    }

    const ndim = view.getUint32(offset, true);
    offset += 4;

    const dtypeCode = view.getInt32(offset, true);
    offset += 4;

    const shape: number[] = [];
    let numElements = 1;
    for (let d = 0; d < ndim; d++) {
      let dimSize: number;
      if (tensorMagic === 0xf993fac9) {
        dimSize = Number(view.getBigInt64(offset, true));
        offset += 8;
      } else {
        dimSize = view.getUint32(offset, true);
        offset += 4;
      }
      shape.push(dimSize);
      numElements *= dimSize;
    }

    let dataView: Float32Array | Float64Array | Int32Array;
    if (dtypeCode === 0) {
      dataView = new Float32Array(buffer.buffer, buffer.byteOffset + offset, numElements);
      offset += numElements * 4;
    } else if (dtypeCode === 1) {
      dataView = new Float64Array(buffer.buffer, buffer.byteOffset + offset, numElements);
      offset += numElements * 8;
    } else if (dtypeCode === 4) {
      dataView = new Int32Array(buffer.buffer, buffer.byteOffset + offset, numElements);
      offset += numElements * 4;
    } else {
      throw new ConversionError(`Unsupported MXNet dtype code: ${dtypeCode}`);
    }

    arrays.push({
      shape,
      data: dataView,
      dtype: dtypeCode,
    });
  }

  const numNames = Number(view.getBigUint64(offset, true));
  offset += 8;

  if (numNames !== numArrays) {
    throw new ConversionError(`Names count ${numNames} does not match arrays count ${numArrays}`);
  }

  const result: Record<string, MxNetParamTensor> = {};
  const decoder = new TextDecoder();
  for (let i = 0; i < numNames; i++) {
    const len = Number(view.getBigUint64(offset, true));
    offset += 8;
    const nameStrBytes = new Uint8Array(buffer.buffer, buffer.byteOffset + offset, len);
    offset += len;
    const name = decoder.decode(nameStrBytes);
    const arr = arrays[i];
    if (arr !== undefined) {
      result[name] = arr;
    }
  }

  return result;
}
