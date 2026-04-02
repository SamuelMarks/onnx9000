import { Graph, ValueInfo } from '../ir/graph.js';
import { Node, Attribute, AttributeType, AttributeValue } from '../ir/node.js';
import { Tensor, Shape, DType } from '../ir/tensor.js';
import {
  Reader,
  readVarInt,
  readVarInt64,
  readTag,
  skipField,
  readString,
  WIRE_TYPE_LENGTH_DELIMITED,
} from './protobuf.js';

/**
 * Maps ONNX enum values to IR DType.
 * @param dataType ONNX enum value representing the data type.
 * @returns The corresponding internal DType string.
 * @throws Error if the data type is not supported.
 */
function mapDataType(dataType: number): DType {
  switch (dataType) {
    case 1:
      return 'float32';
    case 2:
      return 'uint8';
    case 3:
      return 'int8';
    case 4:
      return 'uint16';
    case 5:
      return 'int16';
    case 6:
      return 'int32';
    case 7:
      return 'int64';
    case 8:
      return 'string';
    case 9:
      return 'bool';
    case 10:
      return 'float16';
    case 11:
      return 'float64';
    case 12:
      return 'uint32';
    case 13:
      return 'uint64';
    case 16:
      return 'bfloat16';
    default:
      throw new Error(`Unsupported tensor data type: ${dataType}`);
  }
}

/**
 * Maps ONNX enum values to AttributeType.
 * @param type ONNX enum value representing the attribute type.
 * @returns The corresponding internal AttributeType string.
 */
function mapAttributeType(type: number): AttributeType {
  switch (type) {
    case 1:
      return 'FLOAT';
    case 2:
      return 'INT';
    case 3:
      return 'STRING';
    case 4:
      return 'TENSOR';
    case 5:
      return 'GRAPH';
    case 6:
      return 'FLOATS';
    case 7:
      return 'INTS';
    case 8:
      return 'STRINGS';
    case 9:
      return 'TENSORS';
    case 10:
      return 'GRAPHS';
    case 11:
      return 'SPARSE_TENSOR';
    case 12:
      return 'SPARSE_TENSORS';
    default:
      return 'UNKNOWN';
  }
}

/**
 * Parses an ONNX ModelProto from a Reader.
 * @param reader A protobuf Reader instance containing the model data.
 * @returns A Promise that resolves to the parsed Graph representation.
 */
export async function parseModelProto(reader: Reader): Promise<Graph> {
  const end = reader.getLength();
  const graph = new Graph('');

  try {
    while (reader.getPosition() < end) {
      const tag = await readTag(reader);
      if (tag.fieldNumber === 7) {
        // graph
        const length = await readVarInt(reader);
        const limit = reader.getPosition() + length;
        await parseGraphProto(reader, limit, graph);
      } else if (tag.fieldNumber === 8) {
        // opset_import
        const length = await readVarInt(reader);
        const limit = reader.getPosition() + length;
        let domain = '';
        let version = 1;
        while (reader.getPosition() < limit) {
          const innerTag = await readTag(reader);
          if (innerTag.fieldNumber === 1) {
            // domain
            const strLength = await readVarInt(reader);
            domain = await readString(reader, strLength);
          } else if (innerTag.fieldNumber === 2) {
            // version
            version = await readVarInt(reader);
          } else {
            await skipField(reader, innerTag.wireType);
          }
        }
        graph.opsetImports[domain || ''] = version;
      } else if (tag.fieldNumber === 2) {
        // producer_name
        const length = await readVarInt(reader);
        graph.producerName = await readString(reader, length);
      } else if (tag.fieldNumber === 3) {
        // producer_version
        const length = await readVarInt(reader);
        graph.producerVersion = await readString(reader, length);
      } else if (tag.fieldNumber === 4) {
        // domain
        const length = await readVarInt(reader);
        graph.domain = await readString(reader, length);
      } else if (tag.fieldNumber === 5) {
        // model_version
        graph.modelVersion = await readVarInt(reader);
      } else if (tag.fieldNumber === 6) {
        // doc_string
        const length = await readVarInt(reader);
        graph.docString = await readString(reader, length);
      } else {
        await skipField(reader, tag.wireType);
      }
    }
  } catch (e) {
    console.warn(
      'Model parsing failed partially due to corruption/truncation, returning partial graph:',
      e,
    );
  }

  return graph;
}

/**
 * Parses an ONNX GraphProto.
 * @param reader A protobuf Reader instance.
 * @param limit The end position for this graph proto.
 * @param graph The Graph object to populate with parsed nodes and tensors.
 */
async function parseGraphProto(reader: Reader, limit: number, graph: Graph): Promise<void> {
  try {
    while (reader.getPosition() < limit) {
      const tag = await readTag(reader);
      switch (tag.fieldNumber) {
        case 1: // node
          const nodeLen = await readVarInt(reader);
          const node = await parseNodeProto(reader, reader.getPosition() + nodeLen);
          graph.addNode(node);
          break;
        case 2: // name
          const nameLen = await readVarInt(reader);
          graph.name = await readString(reader, nameLen);
          break;
        case 5: // initializer
          const initLen = await readVarInt(reader);
          const tensor = await parseTensorProto(reader, reader.getPosition() + initLen, true);
          graph.addTensor(tensor);
          graph.initializers.push(tensor.name);
          break;
        case 11: // input
          const inLen = await readVarInt(reader);
          const inInfo = await parseValueInfoProto(reader, reader.getPosition() + inLen);
          graph.inputs.push(inInfo);
          break;
        case 12: // output
          const outLen = await readVarInt(reader);
          const outInfo = await parseValueInfoProto(reader, reader.getPosition() + outLen);
          graph.outputs.push(outInfo);
          break;
        default:
          await skipField(reader, tag.wireType);
      }
    }
  } catch (e) {
    console.warn('Partial graph parsed due to truncation:', e);
  }
}

/**
 * Parses an ONNX NodeProto.
 * @param reader A protobuf Reader instance.
 * @param limit The end position for this node proto.
 * @returns A Promise that resolves to the parsed Node.
 */
async function parseNodeProto(reader: Reader, limit: number): Promise<Node> {
  const inputs: string[] = [];
  const outputs: string[] = [];
  let name = '';
  let opType = '';
  let domain = '';
  let docString = '';
  const attributes: Record<string, Attribute> = {};

  while (reader.getPosition() < limit) {
    const tag = await readTag(reader);
    switch (tag.fieldNumber) {
      case 1: // input
        const inLen = await readVarInt(reader);
        inputs.push(await readString(reader, inLen));
        break;
      case 2: // output
        const outLen = await readVarInt(reader);
        outputs.push(await readString(reader, outLen));
        break;
      case 3: // name
        const nameLen = await readVarInt(reader);
        name = await readString(reader, nameLen);
        break;
      case 4: // op_type
        const opLen = await readVarInt(reader);
        opType = await readString(reader, opLen);
        break;
      case 7: // domain
        const domLen = await readVarInt(reader);
        domain = await readString(reader, domLen);
        break;
      case 6: // doc_string
        const docLen = await readVarInt(reader);
        docString = await readString(reader, docLen);
        break;
      case 5: // attribute
        const attrLen = await readVarInt(reader);
        const attr = await parseAttributeProto(reader, reader.getPosition() + attrLen);
        attributes[attr.name] = attr;
        break;
      default:
        await skipField(reader, tag.wireType);
    }
  }

  return new Node(opType, inputs, outputs, attributes, name, domain, docString);
}

/**
 * Parses an ONNX ValueInfoProto.
 * @param reader A protobuf Reader instance.
 * @param limit The end position for this ValueInfo proto.
 * @returns A Promise that resolves to the parsed ValueInfo.
 */
async function parseValueInfoProto(reader: Reader, limit: number): Promise<ValueInfo> {
  let name = '';
  const shape: Shape = [];
  let dtype: DType = 'float32';

  while (reader.getPosition() < limit) {
    const tag = await readTag(reader);
    switch (tag.fieldNumber) {
      case 1: // name
        const nameLen = await readVarInt(reader);
        name = await readString(reader, nameLen);
        break;
      case 2: // type (TypeProto)
        const typeLen = await readVarInt(reader);
        const typeLimit = reader.getPosition() + typeLen;
        while (reader.getPosition() < typeLimit) {
          const typeTag = await readTag(reader);
          if (typeTag.fieldNumber === 1) {
            // tensor_type
            const tLen = await readVarInt(reader);
            const tLimit = reader.getPosition() + tLen;
            while (reader.getPosition() < tLimit) {
              const tTag = await readTag(reader);
              if (tTag.fieldNumber === 1) {
                // elem_type
                dtype = mapDataType(await readVarInt(reader));
              } else if (tTag.fieldNumber === 2) {
                // shape (TensorShapeProto)
                const sLen = await readVarInt(reader);
                const sLimit = reader.getPosition() + sLen;
                while (reader.getPosition() < sLimit) {
                  const sTag = await readTag(reader);
                  if (sTag.fieldNumber === 1) {
                    // dim
                    const dLen = await readVarInt(reader);
                    const dLimit = reader.getPosition() + dLen;
                    let dimVal: number | string = -1;
                    while (reader.getPosition() < dLimit) {
                      const dTag = await readTag(reader);
                      if (dTag.fieldNumber === 1) {
                        // dim_value
                        dimVal = await readVarInt(reader);
                      } else if (dTag.fieldNumber === 2) {
                        // dim_param
                        const dpLen = await readVarInt(reader);
                        dimVal = await readString(reader, dpLen);
                      } else {
                        await skipField(reader, dTag.wireType);
                      }
                    }
                    shape.push(dimVal);
                  } else {
                    await skipField(reader, sTag.wireType);
                  }
                }
              } else {
                await skipField(reader, tTag.wireType);
              }
            }
          } else {
            await skipField(reader, typeTag.wireType);
          }
        }
        break;
      default:
        await skipField(reader, tag.wireType);
    }
  }

  return new ValueInfo(name, shape, dtype);
}

/**
 * Parses an ONNX AttributeProto.
 * @param reader A protobuf Reader instance.
 * @param limit The end position for this attribute proto.
 * @returns A Promise that resolves to the parsed Attribute.
 */
async function parseAttributeProto(reader: Reader, limit: number): Promise<Attribute> {
  let name = '';
  let typeNum = 0;
  let value: AttributeValue = null;

  while (reader.getPosition() < limit) {
    const tag = await readTag(reader);
    switch (tag.fieldNumber) {
      case 1: // name
        const nameLen = await readVarInt(reader);
        name = await readString(reader, nameLen);
        break;
      case 20: // type
        typeNum = await readVarInt(reader);
        break;
      case 2: // f
        {
          const buf = await reader.readBytes(4);
          const dv = new DataView(buf.buffer, buf.byteOffset, 4);
          value = dv.getFloat32(0, true);
        }
        break;
      case 3: // i
        value = await readVarInt64(reader);
        break;
      case 4: // s
        {
          const sLen = await readVarInt(reader);
          value = await readString(reader, sLen);
        }
        break;
      case 5: // t
        {
          const tLen = await readVarInt(reader);
          value = await parseTensorProto(reader, reader.getPosition() + tLen, false);
        }
        break;
      case 6: // g
        {
          const gLen = await readVarInt(reader);
          const g = new Graph('');
          await parseGraphProto(reader, reader.getPosition() + gLen, g);
          value = g;
        }
        break;
      case 7: // floats
        {
          const floats: number[] = (value as number[]) || [];
          if (tag.wireType === WIRE_TYPE_LENGTH_DELIMITED) {
            const fLen = await readVarInt(reader);
            const fLimit = reader.getPosition() + fLen;
            while (reader.getPosition() < fLimit) {
              const buf = await reader.readBytes(4);
              const dv = new DataView(buf.buffer, buf.byteOffset, 4);
              floats.push(dv.getFloat32(0, true));
            }
          } else {
            const buf = await reader.readBytes(4);
            const dv = new DataView(buf.buffer, buf.byteOffset, 4);
            floats.push(dv.getFloat32(0, true));
          }
          value = floats;
        }
        break;
      case 8: // ints
        {
          const ints: bigint[] = (value as bigint[]) || [];
          if (tag.wireType === WIRE_TYPE_LENGTH_DELIMITED) {
            const iLen = await readVarInt(reader);
            const iLimit = reader.getPosition() + iLen;
            while (reader.getPosition() < iLimit) {
              ints.push(await readVarInt64(reader));
            }
          } else {
            ints.push(await readVarInt64(reader));
          }
          value = ints;
        }
        break;
      case 9: // strings
        {
          const strings: string[] = (value as string[]) || [];
          const strLen = await readVarInt(reader);
          strings.push(await readString(reader, strLen));
          value = strings;
        }
        break;
      case 10: // tensors
        {
          const tensors: Tensor[] = (value as Tensor[]) || [];
          const tensLen = await readVarInt(reader);
          tensors.push(await parseTensorProto(reader, reader.getPosition() + tensLen, false));
          value = tensors;
        }
        break;
      case 11: // graphs
        {
          const graphs: Graph[] = (value as Graph[]) || [];
          const grLen = await readVarInt(reader);
          const gr = new Graph('');
          await parseGraphProto(reader, reader.getPosition() + grLen, gr);
          graphs.push(gr);
          value = graphs;
        }
        break;
      default:
        console.warn(
          `[onnx9000:parser] Unsupported attribute field number ${tag.fieldNumber} encountered on attribute '${name}'.`,
        );
        await skipField(reader, tag.wireType);
        break;
    }
  }

  return new Attribute(name, mapAttributeType(typeNum), value);
}

/**
 * Parses an ONNX TensorProto.
 * @param reader A protobuf Reader instance.
 * @param limit The end position for this tensor proto.
 * @param isInitializer Whether this tensor is an initializer (part of the model weights).
 * @returns A Promise that resolves to the parsed Tensor.
 */
async function parseTensorProto(
  reader: Reader,
  limit: number,
  isInitializer: boolean,
): Promise<Tensor> {
  let name = '';
  const dims: number[] = [];
  let dataTypeNum = 0;
  let rawData: Uint8Array | null = null;
  let dataLocation = 0;
  const externalDataMap: Record<string, string> = {};

  while (reader.getPosition() < limit) {
    const tag = await readTag(reader);
    switch (tag.fieldNumber) {
      case 1: // dims (repeated int64)
        if (tag.wireType === WIRE_TYPE_LENGTH_DELIMITED) {
          const pLen = await readVarInt(reader);
          const pLimit = reader.getPosition() + pLen;
          while (reader.getPosition() < pLimit) {
            dims.push(Number(await readVarInt(reader)));
          }
        } else {
          dims.push(Number(await readVarInt(reader)));
        }
        break;
      case 2: // data_type
        dataTypeNum = await readVarInt(reader);
        break;
      case 8: // name
        const nameLen = await readVarInt(reader);
        name = await readString(reader, nameLen);
        break;
      case 9: // raw_data
        const rawLen = await readVarInt(reader);
        if (rawLen > 0) {
          rawData = await reader.readBytes(rawLen);
        } else {
          rawData = new Uint8Array(0);
        }
        break;
      case 14: // data_location
        dataLocation = await readVarInt(reader);
        break;
      case 13: // external_data
        const extLen = await readVarInt(reader);
        const extLimit = reader.getPosition() + extLen;
        let extKey = '';
        let extVal = '';
        while (reader.getPosition() < extLimit) {
          const innerTag = await readTag(reader);
          if (innerTag.fieldNumber === 1) {
            // key
            const kl = await readVarInt(reader);
            extKey = await readString(reader, kl);
          } else if (innerTag.fieldNumber === 2) {
            // value
            const vl = await readVarInt(reader);
            extVal = await readString(reader, vl);
          } else {
            await skipField(reader, innerTag.wireType);
          }
        }
        if (extKey) {
          externalDataMap[extKey] = extVal;
        }
        break;
      default:
        await skipField(reader, tag.wireType);
    }
  }

  const dtype = mapDataType(dataTypeNum);

  let externalData;
  if (dataLocation === 1 || externalDataMap['location']) {
    // EXTERNAL
    externalData = {
      location: externalDataMap['location'] || '',
      offset: parseInt(externalDataMap['offset'] || '0', 10),
      length: parseInt(externalDataMap['length'] || '0', 10),
    };
  }

  return new Tensor(name, dims, dtype, isInitializer, false, rawData, externalData);
}

/**
 * Explicitly releases an ArrayBuffer reference by nulling it out.
 * @param buffer The ArrayBuffer to release.
 */
export function releaseArrayBuffer(buffer: ArrayBuffer | null): void {
  if (buffer) {
    // Note: This only nulls the local reference; actual release depends on GC.
    buffer = null;
  }
}
