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
  dict: Record<string, ReturnType<typeof JSON.parse>>,
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
  abstract parse(model: ReturnType<typeof JSON.parse>): Graph;
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

  parse(model: ReturnType<typeof JSON.parse>): Graph {
    const graph = new Graph('PyTorch_Exported');
    if (!model || typeof model !== 'object') return graph;

    const fx = model as Record<string, ReturnType<typeof JSON.parse>>;
    if (Array.isArray(fx.nodes)) {
      for (const node of fx.nodes) {
        if (!node || typeof node !== 'object') continue;
        const n = node as Record<string, ReturnType<typeof JSON.parse>>;
        const targetStr = typeof n.target === 'string' ? n.target : '';
        const opStr = typeof n.op === 'string' ? n.op : '';
        const nameStr = typeof n.name === 'string' ? n.name : '';
        const opType = (targetStr || opStr).replace('aten.', '').replace('.default', '');
        const inputs = Array.isArray(n.args)
          ? n.args.filter((arg): arg is string => typeof arg === 'string')
          : [];
        const outputs = [nameStr];
        const rawKwargs =
          typeof n.kwargs === 'object' && n.kwargs !== null
            ? (n.kwargs as Record<string, ReturnType<typeof JSON.parse>>)
            : {};
        const kwargs = mapDictToAttributes(rawKwargs);
        const newNode = new Node(opType, inputs, outputs, kwargs, nameStr);
        graph.nodes.push(newNode);

        const t = new Tensor(nameStr, [], 'float32');
        graph.tensors[nameStr] = t;
      }
    }
    return graph;
  }
}

export class JAXprParser extends BaseParser {
  /** Intercepts JAX closed-form jaxpr representation and maps to GraphSurgeon IR. */

  private mapJaxType(jaxType: string): DType {
    if (jaxType === 'f32') return 'float32';
    if (jaxType === 'i32') return 'int32';
    return 'float32';
  }

  parse(model: ReturnType<typeof JSON.parse>): Graph {
    const graph = new Graph('JAX_Exported');
    if (!model || typeof model !== 'object') return graph;

    const jaxpr = model as Record<string, ReturnType<typeof JSON.parse>>;

    if (Array.isArray(jaxpr.invars)) {
      for (const invar of jaxpr.invars) {
        if (!invar || typeof invar !== 'object') continue;
        const i = invar as Record<string, ReturnType<typeof JSON.parse>>;
        const nameStr = typeof i.name === 'string' ? i.name : '';
        const typeStr = typeof i.type === 'string' ? i.type : '';
        const shapeArr = Array.isArray(i.shape) ? (i.shape as number[]) : [];
        const t = new Tensor(nameStr, shapeArr, this.mapJaxType(typeStr));
        graph.inputs.push(new ValueInfo(nameStr, shapeArr, this.mapJaxType(typeStr)));
        graph.tensors[nameStr] = t;
      }
    }

    if (Array.isArray(jaxpr.constvars)) {
      for (const constvar of jaxpr.constvars) {
        if (!constvar || typeof constvar !== 'object') continue;
        const c = constvar as Record<string, ReturnType<typeof JSON.parse>>;
        const nameStr = typeof c.name === 'string' ? c.name : '';
        const typeStr = typeof c.type === 'string' ? c.type : '';
        const shapeArr = Array.isArray(c.shape) ? (c.shape as number[]) : [];
        const t = new Tensor(nameStr, shapeArr, this.mapJaxType(typeStr), true);
        graph.initializers.push(nameStr);
        graph.tensors[nameStr] = t;
      }
    }

    if (Array.isArray(jaxpr.eqns)) {
      for (const eqn of jaxpr.eqns) {
        if (!eqn || typeof eqn !== 'object') continue;
        const e = eqn as Record<string, ReturnType<typeof JSON.parse>>;
        const primStr = typeof e.primitive === 'string' ? e.primitive : '';
        const inputs = Array.isArray(e.invars)
          ? e.invars
              .map((i: ReturnType<typeof JSON.parse>) => {
                if (i && typeof i === 'object' && 'name' in i) {
                  const n = (i as Record<string, ReturnType<typeof JSON.parse>>).name;
                  return typeof n === 'string' ? n : '';
                }
                return '';
              })
              .filter(Boolean)
          : [];
        const outputs = Array.isArray(e.outvars)
          ? e.outvars
              .map((o: ReturnType<typeof JSON.parse>) => {
                if (o && typeof o === 'object' && 'name' in o) {
                  const n = (o as Record<string, ReturnType<typeof JSON.parse>>).name;
                  return typeof n === 'string' ? n : '';
                }
                return '';
              })
              .filter(Boolean)
          : [];
        const firstOutput = outputs[0] ?? '';
        const nodeName = outputs.length > 0 ? `${primStr}_${firstOutput}` : primStr;
        const rawParams =
          typeof e.params === 'object' && e.params !== null
            ? (e.params as Record<string, ReturnType<typeof JSON.parse>>)
            : {};
        const params = mapDictToAttributes(rawParams);
        const n = new Node(primStr, inputs, outputs, params, nodeName);
        graph.nodes.push(n);

        if (Array.isArray(e.outvars)) {
          for (const outvar of e.outvars) {
            if (!outvar || typeof outvar !== 'object') continue;
            const o = outvar as Record<string, ReturnType<typeof JSON.parse>>;
            const outNameStr = typeof o.name === 'string' ? o.name : '';
            const outTypeStr = typeof o.type === 'string' ? o.type : '';
            const outShapeArr = Array.isArray(o.shape) ? (o.shape as number[]) : [];
            const t = new Tensor(outNameStr, outShapeArr, this.mapJaxType(outTypeStr));
            graph.tensors[outNameStr] = t;
          }
        }
      }
    }

    if (Array.isArray(jaxpr.outvars)) {
      for (const outvar of jaxpr.outvars) {
        if (!outvar || typeof outvar !== 'object') continue;
        const o = outvar as Record<string, ReturnType<typeof JSON.parse>>;
        const nameStr = typeof o.name === 'string' ? o.name : '';
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

  parse(model: ReturnType<typeof JSON.parse>): Graph {
    // Note: XLAHLOParser mock ignores model
    if (model) {
      /* ignored */
    }
    const graph = new Graph('converted_graph');
    graph.name = 'XLA_Exported';
    return graph;
  }
}
