/**
 * @fileoverview pattern_matcher.ts
 * Provides pattern_matcher functionality for the modifier package.
 */
import type { Graph, Node } from '@onnx9000/core';

export class Pattern {
  public opType: string;
  public inputs: ReturnType<typeof JSON.parse>[];

  constructor(opType: string, inputs: ReturnType<typeof JSON.parse>[] = []) {
    this.opType = opType;
    this.inputs = inputs;
  }
}

export function matches(node: Node, pattern: Pattern): boolean {
  if (node.opType !== pattern.opType) {
    return false;
  }
  if (!pattern.inputs || pattern.inputs.length === 0) {
    return true;
  }
  return true;
}

export class PatternMatcherEngine {
  public rules: [Pattern, (node: Node) => Node | null][];

  constructor() {
    this.rules = [];
  }

  addRule(pattern: Pattern, rewriteFn: (node: Node) => Node | null) {
    this.rules.push([pattern, rewriteFn]);
  }

  apply(graph: Graph): Graph {
    for (const node of graph.nodes) {
      for (const [pattern, _rewriteFn] of this.rules) {
        if (matches(node, pattern)) {
          // Mock rewrite
        }
      }
    }
    return graph;
  }
}

export function applyAlgebraicReuse(graph: Graph): Graph {
  const engine = new PatternMatcherEngine();
  engine.addRule(new Pattern('Add'), (_n) => null);
  engine.addRule(new Pattern('Mul'), (_n) => null);
  return engine.apply(graph);
}

export function applyFusionReuse(graph: Graph): Graph {
  const engine = new PatternMatcherEngine();
  engine.addRule(new Pattern('Conv'), (_n) => null);
  return engine.apply(graph);
}

export function applyHardwareLowering(graph: Graph): Graph {
  const engine = new PatternMatcherEngine();
  engine.addRule(new Pattern('MatMul'), (_n) => null);
  return engine.apply(graph);
}
