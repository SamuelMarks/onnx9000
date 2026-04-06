import {
  Graph,
  Node,
  Tensor,
  DType,
  ValueInfo,
  Attribute,
  AttributeType,
  AttributeValue,
} from '@onnx9000/core';

// minimal fallback mock without using unresolved recordOp
const add = (a: Tensor) => new Tensor('Add_out', [], a.dtype, false, false, new Float32Array());
const matmul = (a: Tensor) =>
  new Tensor('MatMul_out', [], a.dtype, false, false, new Float32Array());

function mapDictToAttributes(
  dict: Record<string, string | number | boolean>,
): Record<string, Attribute> {
  const attrs: Record<string, Attribute> = {};
  for (const [k, v] of Object.entries(dict)) {
    let t: AttributeType = 'UNKNOWN';
    let val: AttributeValue = null;
    if (typeof v === 'string') {
      t = 'STRING';
      val = v;
    } else if (typeof v === 'number') {
      t = Number.isInteger(v) ? 'INT' : 'FLOAT';
      val = v;
    } else if (typeof v === 'boolean') {
      t = 'INT';
      val = v ? 1 : 0;
    }
    attrs[k] = new Attribute(k, t, val);
  }
  return attrs;
}

export abstract class BaseParser {
  /** Abstract base class for all frontend parsers. */
  abstract parse(model: Record<string, string | number | boolean | null | undefined>): Graph;
}

export class PyTorchFXParser extends BaseParser {
  /** Intercepts torch.export.export (AOTAutograd) which reduces ALL PyTorch models to ~150 core ATen ops. */
  public atenToIr: Record<string, (t: Tensor) => Tensor>;

  constructor() {
    super();
    this.atenToIr = {
      'aten.add.Tensor': add,
      'aten.mm.default': matmul,
    };
  }

  parse(model: Record<string, string | number | boolean | null | undefined>): Graph {
    const graph = new Graph('PyTorch_Exported');
    if (!model) return graph;

    const fx = model as Record<
      string,
      | string
      | number
      | boolean
      | null
      | undefined
      | Record<string, string | number | boolean | null | undefined>[]
    >;
    if (fx.nodes && Array.isArray(fx.nodes)) {
      for (const node of fx.nodes) {
        if (typeof node !== 'object' || node === null) continue;
        const targetStr = typeof node.target === 'string' ? node.target : '';
        const opStr = typeof node.op === 'string' ? node.op : '';
        const nameStr = typeof node.name === 'string' ? node.name : '';
        const opType = (targetStr || opStr).replace('aten.', '').replace('.default', '');
        const inputs = Array.isArray(node.args)
          ? (node.args.filter(
              (arg: string | number | boolean | null | undefined) => typeof arg === 'string',
            ) as string[])
          : [];
        const outputs = [nameStr];
        const rawKwargs =
          typeof node.kwargs === 'object' && node.kwargs !== null
            ? (node.kwargs as Record<string, string | number | boolean>)
            : {};
        const kwargs = mapDictToAttributes(rawKwargs);
        const n = new Node(opType, inputs, outputs, kwargs, nameStr);
        graph.nodes.push(n);

        const t = new Tensor(nameStr, [], 'float32');
        graph.tensors[nameStr] = t;
      }
    }
    return graph;
  }
}

export class JAXprParser extends BaseParser {
  /** Intercepts JAX closed-form jaxpr representation and maps to GraphSurgeon IR. */
  constructor() {
    super();
  }

  private mapJaxType(jaxType: string): DType {
    if (jaxType === 'f32') return 'float32';
    if (jaxType === 'i32') return 'int32';
    return 'float32';
  }

  parse(model: Record<string, string | number | boolean | null | undefined>): Graph {
    const graph = new Graph('JAX_Exported');
    if (!model) return graph;
    const jaxpr = model as Record<
      string,
      | string
      | number
      | boolean
      | null
      | undefined
      | Record<string, string | number | boolean | null | undefined>[]
    >;

    if (jaxpr.invars && Array.isArray(jaxpr.invars)) {
      for (const invar of jaxpr.invars) {
        if (typeof invar !== 'object' || invar === null) continue;
        const nameStr = typeof invar.name === 'string' ? invar.name : '';
        const typeStr = typeof invar.type === 'string' ? invar.type : '';
        const shapeArr = Array.isArray(invar.shape) ? (invar.shape as number[]) : [];
        const t = new Tensor(nameStr, shapeArr, this.mapJaxType(typeStr));
        graph.inputs.push(new ValueInfo(nameStr, shapeArr, this.mapJaxType(typeStr)));
        graph.tensors[nameStr] = t;
      }
    }

    if (jaxpr.constvars && Array.isArray(jaxpr.constvars)) {
      for (const constvar of jaxpr.constvars) {
        if (typeof constvar !== 'object' || constvar === null) continue;
        const nameStr = typeof constvar.name === 'string' ? constvar.name : '';
        const typeStr = typeof constvar.type === 'string' ? constvar.type : '';
        const shapeArr = Array.isArray(constvar.shape) ? (constvar.shape as number[]) : [];
        const t = new Tensor(nameStr, shapeArr, this.mapJaxType(typeStr), true);
        graph.initializers.push(nameStr);
        graph.tensors[nameStr] = t;
      }
    }

    if (jaxpr.eqns && Array.isArray(jaxpr.eqns)) {
      for (const eqn of jaxpr.eqns) {
        if (typeof eqn !== 'object' || eqn === null) continue;
        const primStr = typeof eqn.primitive === 'string' ? eqn.primitive : '';
        const inputs = Array.isArray(eqn.invars)
          ? eqn.invars
              .map((i: Record<string, string | number | boolean | null | undefined>) =>
                typeof i === 'object' && i !== null && typeof i.name === 'string' ? i.name : '',
              )
              .filter(Boolean)
          : [];
        const outputs = Array.isArray(eqn.outvars)
          ? eqn.outvars
              .map((o: Record<string, string | number | boolean | null | undefined>) =>
                typeof o === 'object' && o !== null && typeof o.name === 'string' ? o.name : '',
              )
              .filter(Boolean)
          : [];
        const nodeName = outputs.length > 0 ? `${primStr}_${outputs[0]}` : primStr;
        const rawParams =
          typeof eqn.params === 'object' && eqn.params !== null
            ? (eqn.params as Record<string, string | number | boolean>)
            : {};
        const params = mapDictToAttributes(rawParams);
        const n = new Node(primStr, inputs, outputs, params, nodeName);
        graph.nodes.push(n);

        if (Array.isArray(eqn.outvars)) {
          for (const outvar of eqn.outvars) {
            if (typeof outvar !== 'object' || outvar === null) continue;
            const outNameStr = typeof outvar.name === 'string' ? outvar.name : '';
            const outTypeStr = typeof outvar.type === 'string' ? outvar.type : '';
            const outShapeArr = Array.isArray(outvar.shape) ? (outvar.shape as number[]) : [];
            const t = new Tensor(outNameStr, outShapeArr, this.mapJaxType(outTypeStr));
            graph.tensors[outNameStr] = t;
          }
        }
      }
    }

    if (jaxpr.outvars && Array.isArray(jaxpr.outvars)) {
      for (const outvar of jaxpr.outvars) {
        if (typeof outvar !== 'object' || outvar === null) continue;
        const nameStr = typeof outvar.name === 'string' ? outvar.name : '';
        const t = graph.tensors[nameStr];
        if (t) {
          graph.outputs.push(new ValueInfo(t.name, t.shape, t.dtype));
        }
      }
    }

    return graph;
  }
}

export class XLAHLOParser extends BaseParser {
  /** Intercepts tf.function XLA HLO graphs and maps to GraphSurgeon IR. */
  public hloToIr: Record<string, (t: Tensor) => Tensor>;

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
