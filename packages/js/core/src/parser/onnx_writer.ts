/* eslint-disable */
import { Graph, ValueInfo } from '../ir/graph.js';
import { Node, Attribute } from '../ir/node.js';
import { Tensor, Shape, DType } from '../ir/tensor.js';
import {
  BufferWriter,
  WIRE_TYPE_LENGTH_DELIMITED,
  WIRE_TYPE_VARINT,
  WIRE_TYPE_32BIT,
  WIRE_TYPE_64BIT,
} from './protobuf_writer.js';

/**
 * Serializes a Graph IR into an ONNX ModelProto bytes.
 * @param graph The computational graph to serialize
 * @returns Serialized ModelProto as Uint8Array
 */
export function serializeModelProto(graph: Graph, opset: number = 14): Uint8Array {
  const writer = new BufferWriter();

  // ir_version (1)
  writer.writeTag(1, WIRE_TYPE_VARINT);
  writer.writeVarInt64(8); // ONNX IR version 8

  // producer_name (2)
  if (graph.producerName) {
    writer.writeTag(2, WIRE_TYPE_LENGTH_DELIMITED);
    writer.writeString(graph.producerName);
  } else {
    writer.writeTag(2, WIRE_TYPE_LENGTH_DELIMITED);
    writer.writeString('onnx9000');
  }

  // opset_import (8)
  const opsetWriter = new BufferWriter();
  opsetWriter.writeTag(1, WIRE_TYPE_LENGTH_DELIMITED);
  opsetWriter.writeString('');
  opsetWriter.writeTag(2, WIRE_TYPE_VARINT);
  opsetWriter.writeVarInt64(opset);
  const opsetBytes = opsetWriter.getResult();
  writer.writeTag(8, WIRE_TYPE_LENGTH_DELIMITED);
  writer.writeVarInt(opsetBytes.length);
  writer.writeBytes(opsetBytes);

  // graph (7)
  const graphWriter = new BufferWriter();

  // name (2)
  graphWriter.writeTag(2, WIRE_TYPE_LENGTH_DELIMITED);
  graphWriter.writeString(graph.name || 'model_graph');

  // nodes (1)
  for (const node of graph.nodes) {
    graphWriter.writeTag(1, WIRE_TYPE_LENGTH_DELIMITED);
    const nodeBytes = serializeNodeProto(node);
    graphWriter.writeVarInt(nodeBytes.length);
    graphWriter.writeBytes(nodeBytes);
  }

  // initializers (5)
  for (const initName of graph.initializers) {
    const tensor = graph.tensors[initName];
    if (tensor) {
      graphWriter.writeTag(5, WIRE_TYPE_LENGTH_DELIMITED);
      const tensorBytes = serializeTensorProto(initName, tensor);
      graphWriter.writeVarInt(tensorBytes.length);
      graphWriter.writeBytes(tensorBytes);
    }
  }

  // inputs (11)
  for (const input of graph.inputs) {
    graphWriter.writeTag(11, WIRE_TYPE_LENGTH_DELIMITED);
    const viBytes = serializeValueInfoProto(input);
    graphWriter.writeVarInt(viBytes.length);
    graphWriter.writeBytes(viBytes);
  }

  // outputs (12)
  for (const output of graph.outputs) {
    graphWriter.writeTag(12, WIRE_TYPE_LENGTH_DELIMITED);
    const viBytes = serializeValueInfoProto(output);
    graphWriter.writeVarInt(viBytes.length);
    graphWriter.writeBytes(viBytes);
  }

  const graphBytes = graphWriter.getResult();
  writer.writeTag(7, WIRE_TYPE_LENGTH_DELIMITED);
  writer.writeVarInt(graphBytes.length);
  writer.writeBytes(graphBytes);

  return writer.getResult();
}

/**
 * Serializes a Node IR into a NodeProto.
 * @param node The node to serialize
 * @returns Serialized NodeProto
 */
function serializeNodeProto(node: Node): Uint8Array {
  const writer = new BufferWriter();

  // inputs (1)
  for (const input of node.inputs) {
    writer.writeTag(1, WIRE_TYPE_LENGTH_DELIMITED);
    writer.writeString(input);
  }

  // outputs (2)
  for (const output of node.outputs) {
    writer.writeTag(2, WIRE_TYPE_LENGTH_DELIMITED);
    writer.writeString(output);
  }

  // name (3)
  if (node.name) {
    writer.writeTag(3, WIRE_TYPE_LENGTH_DELIMITED);
    writer.writeString(node.name);
  }

  // op_type (4)
  if (node.opType) {
    writer.writeTag(4, WIRE_TYPE_LENGTH_DELIMITED);
    writer.writeString(node.opType);
  }

  // attributes (5)
  const attrs: Record<string, Attribute> = node.attributes || {};
  for (const [key, attr] of Object.entries(attrs)) {
    writer.writeTag(5, WIRE_TYPE_LENGTH_DELIMITED);
    const attrBytes = serializeAttributeProto(key, attr);
    writer.writeVarInt(attrBytes.length);
    writer.writeBytes(attrBytes);
  }

  return writer.getResult();
}

/**
 * Serializes an Attribute IR into an AttributeProto.
 * @param name Attribute name
 * @param attr Attribute data
 * @returns Serialized AttributeProto
 */
function serializeAttributeProto(name: string, attr: Attribute): Uint8Array {
  const writer = new BufferWriter();

  writer.writeTag(1, WIRE_TYPE_LENGTH_DELIMITED);
  writer.writeString(name);

  // Type mapping:
  // FLOAT=1, INT=2, STRING=3, TENSOR=4, GRAPH=5
  // FLOATS=6, INTS=7, STRINGS=8, TENSORS=9, GRAPHS=10
  if (attr.type === 'FLOAT') {
    writer.writeTag(20, WIRE_TYPE_VARINT);
    writer.writeVarInt(1);
    writer.writeTag(2, WIRE_TYPE_32BIT);
    writer.writeFloat(attr.value as number);
  } else if (attr.type === 'INT') {
    writer.writeTag(20, WIRE_TYPE_VARINT);
    writer.writeVarInt(2);
    writer.writeTag(3, WIRE_TYPE_VARINT);
    writer.writeVarInt64(attr.value as number);
  } else if (attr.type === 'STRING') {
    writer.writeTag(20, WIRE_TYPE_VARINT);
    writer.writeVarInt(3);
    writer.writeTag(4, WIRE_TYPE_LENGTH_DELIMITED);
    writer.writeString(attr.value as string);
  } else if (attr.type === 'INTS') {
    writer.writeTag(20, WIRE_TYPE_VARINT);
    writer.writeVarInt(7);
    const arr = attr.value as number[];
    for (const v of arr) {
      writer.writeTag(8, WIRE_TYPE_VARINT);
      writer.writeVarInt64(v);
    }
  } else if (attr.type === 'FLOATS') {
    writer.writeTag(20, WIRE_TYPE_VARINT);
    writer.writeVarInt(6);
    const arr = attr.value as number[];
    for (const v of arr) {
      writer.writeTag(7, WIRE_TYPE_32BIT);
      writer.writeFloat(v);
    }
  }

  return writer.getResult();
}

/**
 * Maps IR DType to ONNX TensorProto DataType enum.
 * @param dtype Internal DType
 * @returns ONNX enum value
 */
function mapDTypeToEnum(dtype: DType): number {
  switch (dtype) {
    case 'float32':
      return 1;
    case 'uint8':
      return 2;
    case 'int8':
      return 3;
    case 'uint16':
      return 4;
    case 'int16':
      return 5;
    case 'int32':
      return 6;
    case 'int64':
      return 7;
    case 'string':
      return 8;
    case 'bool':
      return 9;
    case 'float16':
      return 10;
    case 'float64':
      return 11;
    case 'uint32':
      return 12;
    case 'uint64':
      return 13;
    case 'bfloat16':
      return 16;
    default:
      return 1;
  }
}

/**
 * Serializes a Tensor IR into a TensorProto.
 * @param name Tensor name
 * @param tensor Tensor data
 * @returns Serialized TensorProto
 */
function serializeTensorProto(name: string, tensor: Tensor): Uint8Array {
  const writer = new BufferWriter();

  writer.writeTag(8, WIRE_TYPE_LENGTH_DELIMITED);
  writer.writeString(name);

  // data_type (2)
  writer.writeTag(2, WIRE_TYPE_VARINT);
  writer.writeVarInt(mapDTypeToEnum(tensor.dtype));

  // dims (1)
  for (const dim of tensor.shape) {
    writer.writeTag(1, WIRE_TYPE_VARINT);
    writer.writeVarInt64(typeof dim === 'number' ? dim : -1);
  }

  // raw_data (9)
  if (tensor.data) {
    writer.writeTag(9, WIRE_TYPE_LENGTH_DELIMITED);
    const bytes = new Uint8Array(
      tensor.data.buffer,
      tensor.data.byteOffset,
      tensor.data.byteLength,
    );
    writer.writeVarInt(bytes.length);
    writer.writeBytes(bytes);
  }

  return writer.getResult();
}

/**
 * Serializes a ValueInfo IR into a ValueInfoProto.
 * @param vi ValueInfo data
 * @returns Serialized ValueInfoProto
 */
function serializeValueInfoProto(vi: ValueInfo): Uint8Array {
  const writer = new BufferWriter();

  // name (1)
  writer.writeTag(1, WIRE_TYPE_LENGTH_DELIMITED);
  writer.writeString(vi.name);

  // type (2)
  const typeWriter = new BufferWriter();

  // tensor_type (1)
  const tensorTypeWriter = new BufferWriter();

  // elem_type (1)
  tensorTypeWriter.writeTag(1, WIRE_TYPE_VARINT);
  tensorTypeWriter.writeVarInt(mapDTypeToEnum(vi.dtype));

  // shape (2)
  const shapeWriter = new BufferWriter();
  for (const dim of vi.shape) {
    // dim (1)
    const dimWriter = new BufferWriter();
    if (typeof dim === 'number') {
      dimWriter.writeTag(1, WIRE_TYPE_VARINT);
      dimWriter.writeVarInt64(dim);
    } else {
      dimWriter.writeTag(2, WIRE_TYPE_LENGTH_DELIMITED);
      dimWriter.writeString(dim || '?');
    }
    const dimBytes = dimWriter.getResult();
    shapeWriter.writeTag(1, WIRE_TYPE_LENGTH_DELIMITED);
    shapeWriter.writeVarInt(dimBytes.length);
    shapeWriter.writeBytes(dimBytes);
  }

  const shapeBytes = shapeWriter.getResult();
  tensorTypeWriter.writeTag(2, WIRE_TYPE_LENGTH_DELIMITED);
  tensorTypeWriter.writeVarInt(shapeBytes.length);
  tensorTypeWriter.writeBytes(shapeBytes);

  const tensorTypeBytes = tensorTypeWriter.getResult();
  typeWriter.writeTag(1, WIRE_TYPE_LENGTH_DELIMITED);
  typeWriter.writeVarInt(tensorTypeBytes.length);
  typeWriter.writeBytes(tensorTypeBytes);

  const typeBytes = typeWriter.getResult();
  writer.writeTag(2, WIRE_TYPE_LENGTH_DELIMITED);
  writer.writeVarInt(typeBytes.length);
  writer.writeBytes(typeBytes);

  return writer.getResult();
}
