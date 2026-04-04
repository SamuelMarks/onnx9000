import { Graph, Node } from '@onnx9000/core';

export class Pattern {
  public opType: string;
  public inputs: any[];

  constructor(opType: string, inputs: any[] = []) {
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
      for (const [pattern, rewriteFn] of this.rules) {
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
  engine.addRule(new Pattern('Add'), (n) => null);
  engine.addRule(new Pattern('Mul'), (n) => null);
  return engine.apply(graph);
}

export function applyFusionReuse(graph: Graph): Graph {
  const engine = new PatternMatcherEngine();
  engine.addRule(new Pattern('Conv'), (n) => null);
  return engine.apply(graph);
}

export function applyHardwareLowering(graph: Graph): Graph {
  const engine = new PatternMatcherEngine();
  engine.addRule(new Pattern('MatMul'), (n) => null);
  return engine.apply(graph);
}
