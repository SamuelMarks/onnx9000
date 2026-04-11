/* eslint-disable */
import { Node } from './node.js';
import { Tensor, Shape, DType } from './tensor.js';

/**
 * Represents metadata about a value (name, shape, dtype).
 */
export class ValueInfo {
  /** Unique identifier for the value info instance. */
  id: string;
  /** Name of the value. */
  name: string;
  /** Shape of the value. */
  shape: Shape;
  /** Data type of the value. */
  dtype: DType;

  /**
   * Create a new ValueInfo.
   * @param name Name of the value.
   * @param shape Shape of the value.
   * @param dtype Data type of the value.
   */
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

/**
 * Internal Representation of a Model Graph.
 */
export class Graph {
  /** Unique identifier for the graph instance. */
  id: string;
  /** Name of the graph. */
  name: string;
  /** List of nodes in topological order. */
  nodes: Node[];
  /** Map of tensor names to Tensor objects. */
  tensors: Record<string, Tensor>;
  /** List of graph inputs. */
  inputs: ValueInfo[];
  /** List of graph outputs. */
  outputs: ValueInfo[];
  /** List of intermediate value metadata. */
  valueInfo: ValueInfo[];
  /** List of initializer tensor names. */
  initializers: string[];
  /** List of sparse initializer tensor names. */
  sparseInitializers: string[];
  /** Map of opset domains to versions. */
  opsetImports: Record<string, number>;
  /** Name of the graph producer. */
  producerName: string = '';
  /** Version of the graph producer. */
  producerVersion: string = '';
  /** Version of the ONNX model. */
  modelVersion: number = 0;
  /** Domain of the graph. */
  domain: string = '';
  /** Optional documentation string. */
  docString: string = '';
  /** Custom metadata properties. */
  metadataProps: Record<string, string> = {};

  /**
   * Create a new Graph.
   * @param name Name of the graph.
   */
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

  /**
   * Add a tensor to the graph.
   * @param tensor The Tensor object to add.
   */
  addTensor(tensor: Tensor): void {
    this.tensors[tensor.name] = tensor;
  }

  /**
   * Add a node to the graph.
   * @param node The Node object to add.
   */
  addNode(node: Node): void {
    this.nodes.push(node);
  }

  /**
   * Find a node by its name.
   * @param name Name of the node to find.
   * @returns The Node object if found, else null.
   */
  getNode(name: string): Node | null {
    for (const node of this.nodes) {
      if (node.name === name) {
        return node;
      }
    }
    return null;
  }
}
