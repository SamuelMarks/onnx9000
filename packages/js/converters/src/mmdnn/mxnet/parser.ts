/* eslint-disable */
// @ts-nocheck
export interface MxNetSymbol {
  nodes: MxNetNode[];
  arg_nodes: number[];
  heads: [number, number, number][];
}

export interface MxNetNode {
  op: string;
  name: string;
  attrs?: Record<string, string>;
  inputs: [number, number, number][]; // [node_index, output_index, version]
}

export function parseMxNetSymbol(jsonStr: string): MxNetSymbol {
  return JSON.parse(jsonStr) as MxNetSymbol;
}

export function parseMxNetParams(buffer: Uint8Array): Record<string, object> {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  let offset = 0;

  // 1. Read Global Header
  const magic = view.getBigUint64(offset, true);
  offset += 8;

  // Magic 0x0112 is used for standard lists/maps in dmlc
  if (magic !== 274n && magic !== 0x112n) {
    throw new Error(`Invalid MXNet params magic: ${magic.toString(16)}`);
  }

  // Some formats have a reserved 0 uint64
  let count: bigint;
  let nextUint64 = view.getBigUint64(offset, true);
  if (nextUint64 === 0n) {
    offset += 8;
    count = view.getBigUint64(offset, true);
    offset += 8;
  } else {
    count = nextUint64;
    offset += 8;
  }

  const numArrays = Number(count);
  const arrays: object[] = [];

  for (let i = 0; i < numArrays; i++) {
    const tensorMagic = view.getUint32(offset, true);
    offset += 4;

    // v2 magic: 0xF993FAC9 (4187216585)
    // v1 magic: 0x0112 (274)
    if (tensorMagic !== 0xf993fac9 && tensorMagic !== 0x0112) {
      throw new Error(`Invalid MXNet NDArray block magic: ${tensorMagic.toString(16)}`);
    }

    if (tensorMagic === 0xf993fac9) {
      // read reserved 0
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

    // type code mapping
    // 0: float32, 1: float64, 2: float16, 3: uint8, 4: int32, 5: int8, 6: int64
    let dataView: object;
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
      throw new Error(`Unsupported MXNet dtype code: ${dtypeCode}`);
    }

    arrays.push({
      shape,
      data: dataView,
      dtype: dtypeCode,
    });
  }

  // Read names length
  const numNames = Number(view.getBigUint64(offset, true));
  offset += 8;

  if (numNames !== numArrays) {
    throw new Error(`Names count ${numNames} does not match arrays count ${numArrays}`);
  }

  const result: Record<string, object> = {};
  const decoder = new TextDecoder();
  for (let i = 0; i < numNames; i++) {
    const len = Number(view.getBigUint64(offset, true));
    offset += 8;
    const nameStrBytes = new Uint8Array(buffer.buffer, buffer.byteOffset + offset, len);
    offset += len;
    const name = decoder.decode(nameStrBytes);
    result[name] = arrays[i];
  }

  return result;
}
