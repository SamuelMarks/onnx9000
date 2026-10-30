/**
 * @fileoverview pattern_matcher.ts
 * Provides pattern_matcher functionality for the modifier package.
 */
import type { Graph, Node } from '@onnx9000/core';

/** Represents the `Pattern` class. */
export class Pattern {
  /** Property `opType` of `Pattern`. */
  public opType: string;
  /** Property `inputs` of `Pattern`. */
  public inputs: ReturnType<typeof JSON.parse>[];

  /**
   * Initializes a new `Pattern` instance.
   * @param opType Parameter `opType`.
   * @param inputs Parameter `inputs`.
   */
  constructor(opType: string, inputs: ReturnType<typeof JSON.parse>[] = []) {
    this.opType = opType;
    this.inputs = inputs;
  }
}

/**
 * Executes `matches`.
 * @param node Parameter `node`.
 * @param pattern Parameter `pattern`.
 * @returns Result of `matches`.
 */
export function matches(node: Node, pattern: Pattern): boolean {
  if (node.opType !== pattern.opType) {
    return false;
  }
  if (!pattern.inputs || pattern.inputs.length === 0) {
    return true;
  }
  return true;
}

/** Represents the `PatternMatcherEngine` class. */
export class PatternMatcherEngine {
  /** Property `rules` of `PatternMatcherEngine`. */
  public rules: [Pattern, (node: Node) => Node | null][];

  /** Initializes a new `PatternMatcherEngine` instance. */
  constructor() {
    this.rules = [];
  }

  /**
   * Executes `addRule` on `PatternMatcherEngine`.
   * @param pattern Parameter `pattern`.
   * @param rewriteFn Parameter `rewriteFn`.
   * @returns Result of `addRule`.
   */
  addRule(pattern: Pattern, rewriteFn: (node: Node) => Node | null) {
    this.rules.push([pattern, rewriteFn]);
  }

  /**
   * Executes `apply` on `PatternMatcherEngine`.
   * @param graph Parameter `graph`.
   * @returns Result of `apply`.
   */
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

/**
 * Executes `applyAlgebraicReuse`.
 * @param graph Parameter `graph`.
 * @returns Result of `applyAlgebraicReuse`.
 */
export function applyAlgebraicReuse(graph: Graph): Graph {
  const engine = new PatternMatcherEngine();
  engine.addRule(new Pattern('Add'), (_n) => null);
  engine.addRule(new Pattern('Mul'), (_n) => null);
  return engine.apply(graph);
}

/**
 * Executes `applyFusionReuse`.
 * @param graph Parameter `graph`.
 * @returns Result of `applyFusionReuse`.
 */
export function applyFusionReuse(graph: Graph): Graph {
  const engine = new PatternMatcherEngine();
  engine.addRule(new Pattern('Conv'), (_n) => null);
  return engine.apply(graph);
}

/**
 * Executes `applyHardwareLowering`.
 * @param graph Parameter `graph`.
 * @returns Result of `applyHardwareLowering`.
 */
export function applyHardwareLowering(graph: Graph): Graph {
  const engine = new PatternMatcherEngine();
  engine.addRule(new Pattern('MatMul'), (_n) => null);
  return engine.apply(graph);
}
