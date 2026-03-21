export enum MILDataType {
  FLOAT32 = 'fp32',
  FLOAT16 = 'fp16',
  INT32 = 'int32',
  INT64 = 'int64',
  BOOL = 'bool',
  STRING = 'string',
}

export abstract class MILType {
  abstract isTensor(): boolean;
  abstract isScalar(): boolean;
  abstract isTuple(): boolean;
  abstract toString(): string;
}

export class TensorType extends MILType {
  constructor(
    public dataType: MILDataType,
    public shape: (number | string)[],
  ) {
    super();
  }
  isTensor() {
    return true;
  }
  isScalar() {
    return false;
  }
  isTuple() {
    return false;
  }
  toString() {
    return `tensor<${this.dataType}, [${this.shape.join(',')}]>`;
  }
}

export class ScalarType extends MILType {
  constructor(public dataType: MILDataType) {
    super();
  }
  isTensor() {
    return false;
  }
  isScalar() {
    return true;
  }
  isTuple() {
    return false;
  }
  toString() {
    return this.dataType;
  }
}

export class TupleType extends MILType {
  constructor(public elements: MILType[]) {
    super();
  }
  isTensor() {
    return false;
  }
  isScalar() {
    return false;
  }
  isTuple() {
    return true;
  }
  toString() {
    return `tuple<${this.elements.map((e) => e.toString()).join(', ')}>`;
  }
}
