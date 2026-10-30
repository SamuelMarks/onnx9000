/**
 * @fileoverview utilities.ts
 * Provides utilities functionality for the modifier package.
 */
import { Graph, Node, type Tensor } from '@onnx9000/core';
import type { GraphMutator } from '../GraphMutator.js';

/** Represents the `ModifierUtilities` class. */
export class ModifierUtilities {
  /** Property `mutator` of `ModifierUtilities`. */
  mutator: GraphMutator;

  /**
   * Initializes a new `ModifierUtilities` instance.
   * @param mutator Parameter `mutator`.
   */
  constructor(mutator: GraphMutator) {
    this.mutator = mutator;
  }

  // 66. Change Batch Size
  /**
   * Executes `changeBatchSize` on `ModifierUtilities`.
   * @param newBatchSize Parameter `newBatchSize`.
   * @returns Result of `changeBatchSize`.
   */
  changeBatchSize(newBatchSize: number | string) {
    // Collect all unique value info items
    const allItems = [
      ...this.mutator.graph.inputs,
      ...this.mutator.graph.outputs,
      ...this.mutator.graph.valueInfo,
    ];

    // Begin large mutation batch (technically our execute stack pushes one by one,
    // but in a real app we'd want a grouped transaction. We'll mutate the items directly via overrideShape)
    for (const vi of allItems) {
      if (vi.shape && vi.shape.length > 0) {
        const newShape = [...vi.shape];
        newShape[0] = newBatchSize;
        this.mutator.overrideShape(vi.name, newShape, vi.dtype);
      }
    }
  }

  // 67. Make Dynamic
  /**
   * Executes `makeDynamic` on `ModifierUtilities`.
   * @returns Result of `makeDynamic`.
   */
  makeDynamic() {
    this.changeBatchSize('batch_size');
  }

  // 68. Strip Initializers
  /**
   * Executes `stripInitializers` on `ModifierUtilities`.
   * @returns Result of `stripInitializers`.
   */
  stripInitializers() {
    const toRemove = [...this.mutator.graph.initializers];
    for (const init of toRemove) {
      this.mutator.removeInitializer(init);
    }
  }

  // 70. Insert Identity
  /**
   * Executes `insertIdentity` on `ModifierUtilities`.
   * @param edgeName Parameter `edgeName`.
   * @returns Result of `insertIdentity`.
   */
  insertIdentity(edgeName: string) {
    const graph = this.mutator.graph;
    // Find all consumers of this edge
    const consumers = graph.nodes.filter((n) => n.inputs.includes(edgeName));
    if (consumers.length === 0) return; // Nothing to intercept

    const identityOutput = `${edgeName}_identity`;
    this.mutator.addNode('Identity', [edgeName], [identityOutput], {}, `Identity_${edgeName}`);

    // Re-route consumers
    for (const consumer of consumers) {
      for (let i = 0; i < consumer.inputs.length; i++) {
        if (consumer.inputs[i] === edgeName) {
          // 70. Insert Identity
          this.mutator.execute({
            undo: () => {
              consumer.inputs[i] = edgeName;
            },
            redo: () => {
              consumer.inputs[i] = identityOutput;
            },
          });
        }
      }
    }
  }

  // 72. Provide regex-based batch renaming
  /**
   * Executes `regexRenameNodes` on `ModifierUtilities`.
   * @param pattern Parameter `pattern`.
   * @param replacement Parameter `replacement`.
   * @returns Result of `regexRenameNodes`.
   */
  regexRenameNodes(pattern: string, replacement: string) {
    const regex = new RegExp(pattern);
    for (const node of this.mutator.graph.nodes) {
      const oldName = node.name || node.id;
      if (regex.test(oldName)) {
        const newName = oldName.replace(regex, replacement);
        this.mutator.renameNode(oldName, newName);
        node.name = newName;
      }
    }
  }

  // 203. Create a feature to automatically format node names based on depth
  /**
   * Executes `autoFormatNodeNames` on `ModifierUtilities`.
   * @returns Result of `autoFormatNodeNames`.
   */
  autoFormatNodeNames() {
    // Basic topological layer assignment
    const levels = new Map<string, number>();
    for (const init of this.mutator.graph.initializers) levels.set(init, 0);
    for (const input of this.mutator.graph.inputs) levels.set(input.name, 0);

    for (const node of this.mutator.graph.nodes) {
      let maxInputLevel = 0;
      for (const inp of node.inputs) {
        if (inp && levels.has(inp)) {
          maxInputLevel = Math.max(maxInputLevel, levels.get(inp) as number);
        }
      }
      const nodeLevel = maxInputLevel + 1;

      const opLower = node.opType.toLowerCase();
      const newName = `layer_${nodeLevel}/${opLower}`;

      // If we rename it, we must ensure uniqueness, but GraphMutator renameNode does it or we append index
      // GraphMutator renameNode is not strictly unique-enforced unless we do it.
      let uniqueName = newName;
      let counter = 1;
      while (this.mutator.graph.nodes.some((n) => n.name === uniqueName && n !== node)) {
        uniqueName = `${newName}_${counter++}`;
      }

      const oldName = node.name || node.id;
      this.mutator.renameNode(oldName, uniqueName);
      node.name = uniqueName;

      // Assign output levels
      for (const out of node.outputs) {
        if (out) levels.set(out, nodeLevel);
      }
    }
  }

  // 69. Extract Subgraph
  /**
   * Executes `extractSubgraph` on `ModifierUtilities`.
   * @param selectedNodeIds Parameter `selectedNodeIds`.
   * @returns Result of `extractSubgraph`.
   */
  extractSubgraph(selectedNodeIds: string[]): Graph {
    const newGraph = new Graph('ExtractedSubgraph');
    const nodeSet = new Set(selectedNodeIds);
    for (const node of this.mutator.graph.nodes) {
      if (nodeSet.has(node.id)) {
        newGraph.nodes.push(
          new Node(
            node.opType,
            [...node.inputs],
            [...node.outputs],
            { ...node.attributes },
            node.name,
          ),
        );
      }
    }

    for (const vi of this.mutator.graph.valueInfo) {
      newGraph.valueInfo.push(vi);
    }
    for (const t in this.mutator.graph.tensors) {
      newGraph.tensors[t] = this.mutator.graph.tensors[t] as Tensor;
    }
    newGraph.initializers = [...this.mutator.graph.initializers];

    const allInternalOutputs = new Set<string>();
    for (const node of newGraph.nodes) {
      for (const out of node.outputs) allInternalOutputs.add(out);
    }

    const neededInputs = new Set<string>();
    for (const node of newGraph.nodes) {
      for (const inp of node.inputs) {
        if (!allInternalOutputs.has(inp) && inp !== '') {
          neededInputs.add(inp);
        }
      }
    }

    for (const inp of neededInputs) {
      const origInp =
        this.mutator.graph.inputs.find((i) => i.name === inp) ||
        this.mutator.graph.valueInfo.find((v) => v.name === inp);
      if (origInp) {
        newGraph.inputs.push(origInp);
      }
    }

    for (const out of allInternalOutputs) {
      const origOut =
        this.mutator.graph.outputs.find((o) => o.name === out) ||
        this.mutator.graph.valueInfo.find((v) => v.name === out);
      if (origOut) {
        newGraph.outputs.push(origOut);
      }
    }

    return newGraph;
  }

  // 71. Change Opset Version
  // 248. Auto-Fix missing initializers by injecting dummy Zero arrays
  /**
   * Executes `autoFixMissingInitializers` on `ModifierUtilities`.
   * @returns Result of `autoFixMissingInitializers`.
   */
  autoFixMissingInitializers() {
    const unresolvedInputs = new Set<string>();
    const producedEdges = new Set<string>();

    for (const input of this.mutator.graph.inputs) producedEdges.add(input.name);
    for (const init of this.mutator.graph.initializers) producedEdges.add(init);

    for (const node of this.mutator.graph.nodes) {
      for (const out of node.outputs) producedEdges.add(out);
      for (const inp of node.inputs) {
        if (!producedEdges.has(inp) && inp !== '') {
          unresolvedInputs.add(inp);
        }
      }
    }

    if (unresolvedInputs.size === 0) {
      alert('No missing initializers/inputs detected.');
      return;
    }

    for (const missing of Array.from(unresolvedInputs)) {
      // Create a dummy tensor (1 float32)
      const data = new Float32Array([0]);
      this.mutator.addInitializer(missing, 'float32', [1], data);
    }
    alert(`Auto-fixed ${unresolvedInputs.size} missing initializers.`);
  }

  // 225. Validate Opset macro checking compatibility with opset 13-21
  /**
   * Executes `validateOpset` on `ModifierUtilities`.
   * @returns Result of `validateOpset`.
   */
  validateOpset() {
    const aiOnnx =
      this.mutator.graph.opsetImports[''] || this.mutator.graph.opsetImports['ai.onnx'];
    if (aiOnnx && (aiOnnx < 13 || aiOnnx > 21)) {
      alert(
        `Warning: Opset version ${aiOnnx} is outside the officially supported WebNN/WebGPU range (13-21). You may experience compatibility issues.`,
      );
      return false;
    }
    alert(`Opset version ${aiOnnx || 'unknown'} is within the recommended range.`);
    return true;
  }

  /**
   * Executes `changeOpsetVersion` on `ModifierUtilities`.
   * @param domain Parameter `domain`.
   * @param version Parameter `version`.
   * @returns Result of `changeOpsetVersion`.
   */
  changeOpsetVersion(domain: string, version: number) {
    this.mutator.graph.opsetImports[domain] = version;
  }

  // 73. Support injecting Cast nodes automatically if the user connects incompatible types
  /**
   * Executes `injectCastNode` on `ModifierUtilities`.
   * @param edgeName Parameter `edgeName`.
   * @param targetType Parameter `targetType`.
   * @returns Result of `injectCastNode`.
   */
  injectCastNode(edgeName: string, targetType: string) {
    const castNodeName = `Cast_${edgeName}_to_${targetType}`;
    const newEdgeName = `${edgeName}_casted`;
    this.mutator.addNode(
      'Cast',
      [edgeName],
      [newEdgeName],
      {
        to: { name: 'to', type: 'INT', value: targetType === 'FLOAT' ? 1 : 7 },
      },
      castNodeName,
    );
    return newEdgeName;
  }
}
