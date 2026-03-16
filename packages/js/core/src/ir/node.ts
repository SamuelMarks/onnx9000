export type AttributeType =
  | 'FLOAT'
  | 'INT'
  | 'STRING'
  | 'TENSOR'
  | 'GRAPH'
  | 'FLOATS'
  | 'INTS'
  | 'STRINGS'
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
  opType: string;
  inputs: string[];
  outputs: string[];
  attributes: Record<string, Attribute>;
  name: string;
  domain: string;

  constructor(
    opType: string,
    inputs: string[],
    outputs: string[],
    attributes: Record<string, Attribute> = {},
    name: string = '',
    domain: string = '',
  ) {
    this.opType = opType;
    this.inputs = inputs;
    this.outputs = outputs;
    this.attributes = attributes;
    this.name = name;
    this.domain = domain;
  }
}
