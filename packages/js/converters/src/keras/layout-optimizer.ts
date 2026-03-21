/**
 * Represents a single edge between operations detailing the memory layout tensor specification.
 */
export interface LayoutEdge {
  sourceNode: string;
  targetNode: string;
  tensorName: string;
  layout: 'NCHW' | 'NHWC' | 'NCDHW' | 'NDHWC' | 'NCW' | 'NWC' | 'UNKNOWN';
}

/**
 * Interface mapping to an ONNX protobuf Node node equivalent structure.
 */
export interface OnnxNodeLike {
  opType: string;
  attributes: { name: string; ints?: number[] }[];
}

/**
 * A pass that tracks tensor memory layouts globally within a computational graph,
 * selectively eliminating redundant Transpose steps generated dynamically during conversion.
 */
export class LayoutOptimizer {
  /**
   * Collection of recorded tensor edges in topological form.
   */
  public edges: LayoutEdge[] = [];
  private layoutState = new Map<string, LayoutEdge['layout']>();

  /**
   * Records a tensor's memory layout traversing between a source node to a target node.
   *
   * @param tensorName The unique symbolic name of the internal IR tensor.
   * @param sourceNode The identity label of the producer node.
   * @param targetNode The identity label of the consumer node.
   * @param layout The exact explicit layout format, or UNKNOWN.
   */
  public recordEdge(
    tensorName: string,
    sourceNode: string,
    targetNode: string,
    layout: LayoutEdge['layout'],
  ): void {
    this.edges.push({ sourceNode, targetNode, tensorName, layout });
    this.layoutState.set(tensorName, layout);
  }

  /**
   * Retrieves the known tensor memory layout from the tracker context.
   *
   * @param tensorName Internal topological string identifier.
   * @returns The exact enumerated layout structure, or UNKNOWN if missing.
   */
  public getLayout(tensorName: string): LayoutEdge['layout'] {
    return this.layoutState.get(tensorName) || 'UNKNOWN';
  }

  /**
   * Inspects the specified tensor's current data distribution against the target consumer's preferred format.
   *
   * @param tensorName String identifier representing the tensor boundary.
   * @param expectedLayout String identifying the consumer operator's layout specification requirement.
   * @returns A boolean determining if an explicit dynamic transpose needs emitting.
   */
  public needsTranspose(tensorName: string, expectedLayout: LayoutEdge['layout']): boolean {
    const current = this.getLayout(tensorName);
    if (current === 'UNKNOWN' || expectedLayout === 'UNKNOWN') return false;
    return current !== expectedLayout;
  }

  /**
   * Evaluates a sequence of nodes for structural inefficiencies, pruning mutually destructive consecutive operations (e.g., adjacent inverted transposes).
   *
   * @param graphNodes A sequence array representing the ordered AST node structures.
   * @returns An optimized and flattened sequence replacing redundant node boundaries.
   */
  public optimize(graphNodes: OnnxNodeLike[]): OnnxNodeLike[] {
    const optimized: OnnxNodeLike[] = [];
    for (let i = 0; i < graphNodes.length; i++) {
      const node = graphNodes[i];
      if (node!.opType === 'Transpose' && i + 1 < graphNodes.length) {
        const nextNode = graphNodes[i + 1];
        if (nextNode!.opType === 'Transpose') {
          const perm1 = node!.attributes.find((a) => a.name === 'perm')?.ints;
          const perm2 = nextNode!.attributes.find((a) => a.name === 'perm')?.ints;

          if (this.isIdentityPermutation(perm1, perm2)) {
            i++; // skip nextNode too
            continue;
          }
        }
      }
      optimized.push(node!);
    }
    return optimized;
  }

  private isIdentityPermutation(perm1?: number[], perm2?: number[]): boolean {
    if (!perm1 || !perm2 || perm1.length !== perm2.length) return false;
    const combined = new Array(perm1.length);
    for (let i = 0; i < perm1.length; i++) {
      combined[i] = perm1[perm2[i]!];
    }
    for (let i = 0; i < combined.length; i++) {
      if (combined[i] !== i) return false;
    }
    return true;
  }
}
