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

export type AttributeValue = number | string | number[] | string[] | object | null;

export class Attribute {
  name: string;
  type: AttributeType;
  value: AttributeValue;

  constructor(name: string, type: AttributeType, value: AttributeValue) {
    this.name = name;
    this.type = type;
    this.value = value;
  }
}

export class Node {
  id: string;
  opType: string;
  inputs: string[];
  outputs: string[];
  attributes: Record<string, Attribute>;
  name: string;
  domain: string;
  docString: string;

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
