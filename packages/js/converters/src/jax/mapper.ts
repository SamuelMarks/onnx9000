/**
 * Mapper for JAX to ONNX IR.
 */

import { Graph, ValueInfo, Node, Attribute, Tensor, AttributeValue } from '@onnx9000/core';
import { Jaxpr } from './jaxpr_parser.js';

export class JaxMapper {
  jaxpr: Jaxpr;
  flaxState?: Record<string, object>;
  graph: Graph;

  constructor(jaxpr: Jaxpr, flaxState?: Record<string, object>) {
    this.jaxpr = jaxpr;
    if (flaxState) {
      this.flaxState = flaxState;
    }
    this.graph = new Graph('JaxModel');
  }

  getTensor(name: string): Tensor {
    if (!this.graph.tensors[name]) {
      const v = new Tensor(name, [], 'float32');
      this.graph.addTensor(v);
      return v;
    }
    return this.graph.tensors[name];
  }

  map(): Graph {
    // Inputs
    for (const invar of this.jaxpr.invars) {
      this.getTensor(invar);
      this.graph.inputs.push(new ValueInfo(invar, [], 'float32'));
    }

    // Constants/Flax State
    // Basic mapping of state variables if provided
    for (const cvar of this.jaxpr.constvars) {
      if (this.flaxState && this.flaxState[cvar]) {
        const val = this.flaxState[cvar] as number[] | { flat: () => number[] };
        let arr: number[] = [];
        if ('flat' in val && typeof val.flat === 'function') {
          const flatResult = val.flat();
          arr = Array.isArray(flatResult) ? (flatResult as number[]) : [];
        } else if (Array.isArray(val)) {
          arr = val;
        }
        const data = new Float32Array(arr);
        const t = new Tensor(cvar, [], 'float32', true, false, new Uint8Array(data.buffer));
        this.graph.addTensor(t);
      } else {
        this.getTensor(cvar);
        // Treat as input if missing
        this.graph.inputs.push(new ValueInfo(cvar, [], 'float32'));
      }
    }

    // Equations
    for (const eqn of this.jaxpr.eqns) {
      const inputs = eqn.invars.map((i) => this.getTensor(i).name);
      const outputs = eqn.outvars.map((o) => this.getTensor(o).name);

      let opType = eqn.primitive;

      // Simple mapping dictionary
      const opMap: Record<string, string | undefined> = {
        add: 'Add',
        sub: 'Sub',
        mul: 'Mul',
        div: 'Div',
        dot_general: 'MatMul',
        broadcast_in_dim: 'Expand',
        reshape: 'Reshape',
        conv_general_dilated: 'Conv',
        max_pool: 'MaxPool',
        reduce_sum: 'ReduceSum',
      };

      const mappedOp = opMap[eqn.primitive];
      if (mappedOp) {
        opType = mappedOp;
      }

      const node = new Node(opType, inputs, outputs, {}, `${eqn.primitive}_node`);

      // Set some attributes if they exist
      for (const key in eqn.params) {
        const val = eqn.params[key];
        let type:
          | 'FLOAT'
          | 'INT'
          | 'STRING'
          | 'INTS'
          | 'FLOATS'
          | 'TENSOR'
          | 'GRAPH'
          | 'SPARSE_TENSOR'
          | 'TENSORS'
          | 'GRAPHS'
          | 'SPARSE_TENSORS' = 'STRING';
        if (typeof val === 'number') {
          type = Number.isInteger(val) ? 'INT' : 'FLOAT';
        } else if (typeof val === 'string') {
          type = 'STRING';
        } else if (Array.isArray(val)) {
          type =
            typeof val[0] === 'number' ? (Number.isInteger(val[0]) ? 'INTS' : 'FLOATS') : 'STRING';
        }
        node.attributes[key] = new Attribute(key, type, val as AttributeValue);
      }

      this.graph.addNode(node);
    }

    // Outputs
    for (const outvar of this.jaxpr.outvars) {
      this.getTensor(outvar);
      this.graph.outputs.push(new ValueInfo(outvar, [], 'float32'));
    }

    return this.graph;
  }
}
