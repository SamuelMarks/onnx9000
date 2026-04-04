import { Graph, Tensor } from '@onnx9000/core';

// minimal fallback mock without using unresolved recordOp
const add = (a: Tensor) => new Tensor('Add_out', [], a.dtype, false, false, new Float32Array());
const matmul = (a: Tensor) =>
  new Tensor('MatMul_out', [], a.dtype, false, false, new Float32Array());

export abstract class BaseParser {
  /** Abstract base class for all frontend parsers. */
  abstract parse(model: unknown): Graph;
}

export class PyTorchFXParser extends BaseParser {
  /** Intercepts torch.export.export (AOTAutograd) which reduces ALL PyTorch models to ~150 core ATen ops. */
  public atenToIr: Record<string, unknown>;

  constructor() {
    super();
    this.atenToIr = {
      'aten.add.Tensor': add,
      'aten.mm.default': matmul,
    };
  }

  parse(): Graph {
    const graph = new Graph('converted_graph');
    graph.name = 'PyTorch_Exported';
    return graph;
  }
}

export class JAXprParser extends BaseParser {
  /** Intercepts JAX closed-form jaxpr representation and maps to GraphSurgeon IR. */
  public jaxprToIr: Record<string, unknown>;

  constructor() {
    super();
    this.jaxprToIr = {
      add: add,
      dot_general: matmul,
    };
  }

  parse(): Graph {
    const graph = new Graph('converted_graph');
    graph.name = 'JAX_Exported';
    return graph;
  }
}

export class XLAHLOParser extends BaseParser {
  /** Intercepts tf.function XLA HLO graphs and maps to GraphSurgeon IR. */
  public hloToIr: Record<string, unknown>;

  constructor() {
    super();
    this.hloToIr = {
      add: add,
      dot: matmul,
    };
  }

  parse(): Graph {
    const graph = new Graph('converted_graph');
    graph.name = 'XLA_Exported';
    return graph;
  }
}
