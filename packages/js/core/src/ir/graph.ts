import { Node } from './node.js';
import { Tensor, Shape, DType } from './tensor.js';

export class ValueInfo {
  id: string;
  name: string;
  shape: Shape;
  dtype: DType;

  constructor(name: string, shape: Shape, dtype: DType) {
    this.id =
      typeof crypto !== 'undefined' && crypto.randomUUID
        ? crypto.randomUUID()
        : Math.random().toString(36).substring(2);
    this.name = name;
    this.shape = shape;
    this.dtype = dtype;
  }
}

export class Graph {
  id: string;
  name: string;
  nodes: Node[];
  tensors: Record<string, Tensor>;
  inputs: ValueInfo[];
  outputs: ValueInfo[];
  valueInfo: ValueInfo[];
  initializers: string[];
  sparseInitializers: string[];
  opsetImports: Record<string, number>;
  producerName: string = '';
  producerVersion: string = '';
  modelVersion: number = 0;
  domain: string = '';
  docString: string = '';
  metadataProps: Record<string, string> = {};

  constructor(name: string) {
    this.id =
      typeof crypto !== 'undefined' && crypto.randomUUID
        ? crypto.randomUUID()
        : Math.random().toString(36).substring(2);
    this.name = name;
    this.nodes = [];
    this.tensors = {};
    this.inputs = [];
    this.outputs = [];
    this.valueInfo = [];
    this.initializers = [];
    this.sparseInitializers = [];
    this.opsetImports = {};
    this.docString = '';
    this.metadataProps = {};
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
