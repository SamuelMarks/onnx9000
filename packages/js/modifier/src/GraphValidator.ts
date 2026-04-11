/* eslint-disable */
import { Graph, Node, ValueInfo, Tensor, Shape, DType } from '@onnx9000/core';

export interface GraphValidationResult {
  isValid: boolean;
  danglingNodes: string[]; // Node names
  unresolvedInputs: string[]; // Edge names
  cyclicDependencies: string[]; // Node names in cycle
  typeMismatches: string[]; // Details

  missingShapes: string[]; // Inputs lacking defined shape properties
  dimensionMismatches: string[]; // Details
}

/**
 * A class responsible for fast static analysis of the graph.
 * Checks for dangling nodes, type mismatches, and cycles.
 */
export class GraphValidator {
  graph: Graph;

  constructor(graph: Graph) {
    this.graph = graph;
  }

  // 26. Implement a fast, synchronous verify() method
  verify(): GraphValidationResult {
    const result: GraphValidationResult = {
      isValid: true,
      danglingNodes: [],
      unresolvedInputs: [],
      cyclicDependencies: [],
      typeMismatches: [],
      dimensionMismatches: [],
      missingShapes: [],
    };

    const producedEdges = new Set<string>();
    const consumedEdges = new Set<string>();

    for (const input of this.graph.inputs) producedEdges.add(input.name);
    for (const init of this.graph.initializers) producedEdges.add(init);

    for (const node of this.graph.nodes) {
      for (const out of node.outputs) producedEdges.add(out);
      for (const inp of node.inputs) consumedEdges.add(inp);
    }
    for (const out of this.graph.outputs) consumedEdges.add(out.name);

    // 268. Detect inputs lacking defined shape properties
    for (const input of this.graph.inputs) {
      if (!input.shape || input.shape.length === 0) {
        result.missingShapes.push(input.name);
      }
    }

    // 27. Detect dangling nodes
    for (const node of this.graph.nodes) {
      let consumed = false;
      for (const out of node.outputs) {
        if (consumedEdges.has(out) || this.graph.outputs.some((o) => o.name === out)) {
          consumed = true;
          break;
        }
      }
      if (!consumed && node.outputs.length > 0) {
        result.danglingNodes.push(node.name || node.id);
      }
    }

    // 28. Detect unresolved inputs
    for (const node of this.graph.nodes) {
      for (const inp of node.inputs) {
        // empty strings are often used for optional inputs in ONNX
        if (inp !== '' && !producedEdges.has(inp)) {
          result.unresolvedInputs.push(inp);
        }
      }
    }

    // 29. Detect cyclic dependencies (Tarjan's or simple DFS)
    const visited = new Set<string>();
    const recursionStack = new Set<string>();
    const producerMap = new Map<string, Node>();

    for (const node of this.graph.nodes) {
      for (const out of node.outputs) {
        producerMap.set(out, node);
      }
    }

    const checkCycle = (node: Node) => {
      if (recursionStack.has(node.id)) {
        result.cyclicDependencies.push(node.name || node.id);
        return true;
      }
      if (visited.has(node.id)) return false;

      visited.add(node.id);
      recursionStack.add(node.id);

      for (const inp of node.inputs) {
        const prod = producerMap.get(inp);
        if (prod) {
          if (checkCycle(prod)) return true;
        }
      }

      recursionStack.delete(node.id);
      return false;
    };

    for (const node of this.graph.nodes) {
      checkCycle(node);
    }

    // 30 & 33. Mock type & dimension checking (real checking requires full shape inference)
    // For now, we will just flag obvious MatMul mismatches if valueInfo is populated
    const shapes = new Map<string, Shape>();
    const dtypes = new Map<string, DType>();

    for (const vi of this.graph.inputs) {
      shapes.set(vi.name, vi.shape);
      dtypes.set(vi.name, vi.dtype);
    }
    for (const vi of this.graph.valueInfo) {
      shapes.set(vi.name, vi.shape);
      dtypes.set(vi.name, vi.dtype);
    }
    for (const t of Object.values(this.graph.tensors)) {
      shapes.set(t.name, t.shape);
      dtypes.set(t.name, t.dtype);
    }

    for (const node of this.graph.nodes) {
      if (node.opType === 'MatMul' && node.inputs.length === 2) {
        const shapeA = shapes.get(node.inputs[0]!);
        const shapeB = shapes.get(node.inputs[1]!);
        if (shapeA && shapeB && shapeA.length >= 2 && shapeB.length >= 2) {
          const kA = shapeA[shapeA.length - 1];
          const kB = shapeB[shapeB.length - 2];
          if (
            kA !== kB &&
            kA !== -1 &&
            kB !== -1 &&
            typeof kA === 'number' &&
            typeof kB === 'number'
          ) {
            result.dimensionMismatches.push(`MatMul ${node.name}: ${kA} != ${kB}`);
          }
        }

        const typeA = dtypes.get(node.inputs[0]!);
        const typeB = dtypes.get(node.inputs[1]!);
        if (typeA && typeB && typeA !== typeB) {
          result.typeMismatches.push(`MatMul ${node.name}: ${typeA} != ${typeB}`);
        }
      }
    }

    result.isValid =
      result.danglingNodes.length === 0 &&
      result.unresolvedInputs.length === 0 &&
      result.cyclicDependencies.length === 0 &&
      result.typeMismatches.length === 0 &&
      result.missingShapes.length === 0 &&
      result.dimensionMismatches.length === 0;

    return result;
  }
}
