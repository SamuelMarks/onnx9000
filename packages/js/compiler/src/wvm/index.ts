import { Graph } from '@onnx9000/core';

export function emitWVM(graph: Graph): Uint8Array {
  if (graph.nodes.length === 0) {
    throw new Error('Graph is empty');
  }
  return new Uint8Array([0x57, 0x56, 0x4d]); // 'WVM' mock
}

export class WVMInterpreter {
  private bytecode: Uint8Array;
  constructor(bytecode: Uint8Array) {
    this.bytecode = bytecode;
  }
  execute(): boolean {
    if (this.bytecode.length === 0) {
      throw new Error('Empty bytecode');
    }
    return true;
  }
}
