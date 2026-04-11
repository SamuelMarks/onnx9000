/* eslint-disable */
/**
 * Enumeration of ONNX attribute types.
 */
export type AttributeType =
  | 'FLOAT'
  | 'INT'
  | 'STRING'
  | 'TENSOR'
  | 'GRAPH'
  | 'FLOATS'
  | 'INTS'
  | 'STRINGS'
  | 'TENSORS'
  | 'GRAPHS'
  | 'SPARSE_TENSOR'
  | 'SPARSE_TENSORS'
  | 'UNKNOWN';

/**
 * Union of all possible attribute value types.
 */
export type AttributeValue =
  | number
  | bigint
  | string
  | number[]
  | bigint[]
  | string[]
  | object
  | null;

/**
 * Represents an attribute in an ONNX node.
 */
export class Attribute {
  /** The attribute name */
  name: string;
  /** The attribute type */
  type: AttributeType;
  /** The attribute value */
  value: AttributeValue;

  /**
   * Create a new Attribute instance.
   * @param name The attribute name
   * @param type The attribute type
   * @param value The attribute value
   */
  constructor(name: string, type: AttributeType, value: AttributeValue) {
    this.name = name;
    this.type = type;
    this.value = value;
  }
}

/**
 * Represents a node in an ONNX graph.
 */
export class Node {
  /** Unique identifier for the node */
  id: string;
  /** Operation type (e.g., 'Add', 'Conv') */
  opType: string;
  /** List of input tensor names */
  inputs: string[];
  /** List of output tensor names */
  outputs: string[];
  /** Mapping of attribute names to Attribute objects */
  attributes: Record<string, Attribute>;
  /** Node name (optional) */
  name: string;
  /** Domain of the operator (optional, default: 'ai.onnx') */
  domain: string;
  /** Documentation string (optional) */
  docString: string;

  /**
   * Create a new Node instance.
   * @param opType Operation type
   * @param inputs List of input tensor names
   * @param outputs List of output tensor names
   * @param attributes Mapping of attribute names to Attribute objects
   * @param name Node name
   * @param domain Domain of the operator
   * @param docString Documentation string
   */
  constructor(
    opType: string,
    inputs: string[],
    outputs: string[],
    attributes: Record<string, Attribute> = {},
    name: string = '',
    domain: string = '',
    docString: string = '',
  ) {
    this.id =
      typeof crypto !== 'undefined' && crypto.randomUUID
        ? crypto.randomUUID()
        : Math.random().toString(36).substring(2);
    this.opType = opType;
    this.inputs = inputs;
    this.outputs = outputs;
    this.attributes = attributes;
    this.name = name;
    this.domain = domain;
    this.docString = docString;
  }
}
