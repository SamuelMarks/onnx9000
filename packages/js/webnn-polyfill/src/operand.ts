import { MLOperand, MLOperandDataType } from './interfaces.js';

export class PolyfillMLOperand implements MLOperand {
  dataType: MLOperandDataType;
  shape: number[];
  name: string; // The ONNX value name

  constructor(name: string, dataType: MLOperandDataType, shape: number[]) {
    this.name = name;
    this.dataType = dataType;
    this.shape = shape;
  }
}
