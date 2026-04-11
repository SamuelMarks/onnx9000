/* eslint-disable */
/**
 * Topological sorting implementation mapping Kahn's algorithm for MIL graphs.
 * @module
 */
import { Operation } from './ast.js';

/**
 * Topologically sorts an array of independent `mil.Operation` nodes ensuring input validity.
 * Detects cycles within directed acyclic graphs automatically.
 * @param operations - The unordered MIL Operations array.
 * @returns The topologically sorted array of Operations.
 * @throws {Error} if cyclic dependency topologies are detected (e.g. A depends on B, B depends on A).
 */
export function topologicalSort(operations: Operation[]): Operation[] {
  const opMap = new Map<string, Operation>();
  const inDegree = new Map<string, number>();
  const adjList = new Map<string, string[]>();

  // Map each operation to a unique ID (based on output var name)
  for (const op of operations) {
    const opId = op.outputs.map((o) => o.name).join(',');
    opMap.set(opId, op);
    inDegree.set(opId, 0);
    adjList.set(opId, []);
  }

  // Build the graph
  for (const op of operations) {
    const opId = op.outputs.map((o) => o.name).join(',');
    for (const inputKey in op.inputs) {
      const inputs = op.inputs[inputKey]!;
      const inputArray = Array.isArray(inputs) ? inputs : [inputs];

      for (const inputVar of inputArray) {
        // Find which operation produces this input
        for (const [otherId, otherOp] of opMap) {
          if (otherOp.outputs.some((o) => o.name === inputVar.name)) {
            adjList.get(otherId)!.push(opId);
            inDegree.set(opId, inDegree.get(opId)! + 1);
            break;
          }
        }
      }
    }
  }

  // Kahn's algorithm
  const queue: string[] = [];
  for (const [opId, deg] of inDegree) {
    if (deg === 0) queue.push(opId);
  }

  const sorted: Operation[] = [];
  while (queue.length > 0) {
    const currentId = queue.shift()!;
    sorted.push(opMap.get(currentId)!);

    for (const neighbor of adjList.get(currentId)!) {
      inDegree.set(neighbor, inDegree.get(neighbor)! - 1);
      if (inDegree.get(neighbor) === 0) {
        queue.push(neighbor);
      }
    }
  }

  if (sorted.length !== operations.length) {
    throw new Error('Cycle detected in MIL graph operations');
  }

  return sorted;
}
