import { Node } from './node.js';
import { Tensor, Shape, DType } from './tensor.js';

export class ValueInfo {
  name: string;
  shape: Shape;
  dtype: DType;

  constructor(name: string, shape: Shape, dtype: DType) {
    this.name = name;
    this.shape = shape;
    this.dtype = dtype;
  }
}

export class Graph {
  name: string;
  nodes: Node[];
  tensors: Record<string, Tensor>;
  inputs: ValueInfo[];
  outputs: ValueInfo[];
  initializers: string[];
  opsetImports: Record<string, number>;
  docString: string;

  constructor(name: string) {
    this.name = name;
    this.nodes = [];
    this.tensors = {};
    this.inputs = [];
    this.outputs = [];
    this.initializers = [];
    this.opsetImports = {};
    this.docString = '';
  }

  addTensor(tensor: Tensor): void {
    this.tensors[tensor.name] = tensor;
  }

  addNode(node: Node): void {
    this.nodes.push(node);
  }

  getNode(name: string): Node | null {
    for (const node of this.nodes) {
      if (node.name === name) {
        return node;
      }
    }
    return null;
  }
}
