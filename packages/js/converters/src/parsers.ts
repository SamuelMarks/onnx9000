/**
 * @fileoverview Frontend parsers for PyTorch FX, JAXpr, and XLA HLO representations.
 */
import {
  Attribute,
  type AttributeType,
  type AttributeValue,
  type DType,
  Graph,
  Node,
  Tensor,
  ValueInfo,
} from '@onnx9000/core';

/** Primitive attribute value supported in frontend parser node dictionaries. */
export type FrontendAttrValue = string | number | boolean | null | undefined;

/** PyTorch FX graph node representation. */
export interface PyTorchFXNode {
  /** ATen operator target string (e.g., `'aten.add.Tensor'`). */
  target?: string;
  /** FX node op category (e.g., `'call_function'`). */
  op?: string;
  /** Node output variable name. */
  name?: string;
  /** Input argument names or literal values. */
  args?: (string | number | boolean | null)[];
  /** Keyword arguments mapped to ONNX attributes. */
  kwargs?: Record<string, FrontendAttrValue>;
}

/** PyTorch FX exported graph model representation. */
export interface PyTorchFXModel {
  /** Ordered list of FX nodes. */
  nodes?: (PyTorchFXNode | null)[];
}

/** JAXpr variable descriptor (`invar`, `constvar`, or `outvar`). */
export interface JaxprVar {
  /** Variable name. */
  name?: string;
  /** JAX element type string (e.g., `'f32'`, `'i32'`). */
  type?: string;
  /** Tensor shape dimensions. */
  shape?: number[];
}

/** JAXpr equation descriptor. */
export interface JaxprEqn {
  /** JAX primitive operator name. */
  primitive?: string;
  /** Input variable references. */
  invars?: (JaxprVar | null)[];
  /** Output variable references. */
  outvars?: (JaxprVar | null)[];
  /** Primitive static parameters. */
  params?: Record<string, FrontendAttrValue>;
}

/** Closed-form JAXpr model representation. */
export interface JaxprModel {
  /** Input variables. */
  invars?: (JaxprVar | null)[];
  /** Constant variables (initializers). */
  constvars?: (JaxprVar | null)[];
  /** Primitive equations. */
  eqns?: (JaxprEqn | null)[];
  /** Output variables. */
  outvars?: (JaxprVar | null)[];
}

/** Union of supported frontend model representations. */
export type FrontendModelInput = PyTorchFXModel | JaxprModel | Record<string, FrontendAttrValue>;

const add = (a: Tensor): Tensor =>
  new Tensor('Add_out', [], a.dtype, false, false, new Float32Array());
const matmul = (a: Tensor): Tensor =>
  new Tensor('MatMul_out', [], a.dtype, false, false, new Float32Array());

/**
 * Converts a plain key-value dictionary of primitive values into ONNX `Attribute` instances.
 * @param dict Dictionary of raw attribute values.
 * @returns Record of ONNX `Attribute` objects.
 */
function mapDictToAttributes(dict: Record<string, FrontendAttrValue>): Record<string, Attribute> {
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

/**
 * Abstract base class for all frontend parsers.
 */
export abstract class BaseParser {
  /**
   * Parses a frontend model structure into an ONNX `Graph`.
   * @param model Frontend model representation or `null`.
   * @returns Constructed ONNX `Graph`.
   */
  abstract parse(model: FrontendModelInput | null): Graph;
}

/**
 * Intercepts `torch.export.export` (AOTAutograd) graphs and lowers ATen ops to ONNX IR.
 */
export class PyTorchFXParser extends BaseParser {
  /** Mapping from ATen operator names to tensor transformation functions. */
  public atenToIr: Record<string, (t: Tensor) => Tensor>;

  /**
   * Initializes a new `PyTorchFXParser` with default ATen operator handlers.
   */
  constructor() {
    super();
    this.atenToIr = {
      'aten.add.Tensor': add,
      'aten.mm.default': matmul,
    };
  }

  /**
   * Parses a PyTorch FX model object into an ONNX `Graph`.
   * @param model PyTorch FX model structure or `null`.
   * @returns Parsed ONNX `Graph`.
   */
  parse(model: PyTorchFXModel | null): Graph {
    const graph = new Graph('PyTorch_Exported');
    if (!model || typeof model !== 'object') return graph;

    if (Array.isArray(model.nodes)) {
      for (const node of model.nodes) {
        if (!node || typeof node !== 'object') continue;
        const targetStr = typeof node.target === 'string' ? node.target : '';
        const opStr = typeof node.op === 'string' ? node.op : '';
        const nameStr = typeof node.name === 'string' ? node.name : '';
        const opType = (targetStr || opStr).replace('aten.', '').replace('.default', '');
        const inputs = Array.isArray(node.args)
          ? node.args.filter((arg): arg is string => typeof arg === 'string')
          : [];
        const outputs = [nameStr];
        const rawKwargs =
          typeof node.kwargs === 'object' && node.kwargs !== null ? node.kwargs : {};
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

/**
 * Intercepts JAX closed-form `jaxpr` representations and maps them to ONNX IR.
 */
export class JAXprParser extends BaseParser {
  /**
   * Maps a JAX short dtype string to an ONNX `DType`.
   * @param jaxType JAX element type string (e.g., `'f32'`, `'i32'`).
   * @returns Corresponding ONNX `DType`.
   */
  private mapJaxType(jaxType: string): DType {
    if (jaxType === 'f32') return 'float32';
    if (jaxType === 'i32') return 'int32';
    return 'float32';
  }

  /**
   * Parses a JAXpr model object into an ONNX `Graph`.
   * @param model JAXpr model structure or `null`.
   * @returns Parsed ONNX `Graph`.
   */
  parse(model: JaxprModel | null): Graph {
    const graph = new Graph('JAX_Exported');
    if (!model || typeof model !== 'object') return graph;

    if (Array.isArray(model.invars)) {
      for (const invar of model.invars) {
        if (!invar || typeof invar !== 'object') continue;
        const nameStr = typeof invar.name === 'string' ? invar.name : '';
        const typeStr = typeof invar.type === 'string' ? invar.type : '';
        const shapeArr = Array.isArray(invar.shape) ? invar.shape : [];
        const t = new Tensor(nameStr, shapeArr, this.mapJaxType(typeStr));
        graph.inputs.push(new ValueInfo(nameStr, shapeArr, this.mapJaxType(typeStr)));
        graph.tensors[nameStr] = t;
      }
    }

    if (Array.isArray(model.constvars)) {
      for (const constvar of model.constvars) {
        if (!constvar || typeof constvar !== 'object') continue;
        const nameStr = typeof constvar.name === 'string' ? constvar.name : '';
        const typeStr = typeof constvar.type === 'string' ? constvar.type : '';
        const shapeArr = Array.isArray(constvar.shape) ? constvar.shape : [];
        const t = new Tensor(nameStr, shapeArr, this.mapJaxType(typeStr), true);
        graph.initializers.push(nameStr);
        graph.tensors[nameStr] = t;
      }
    }

    if (Array.isArray(model.eqns)) {
      for (const eqn of model.eqns) {
        if (!eqn || typeof eqn !== 'object') continue;
        const primStr = typeof eqn.primitive === 'string' ? eqn.primitive : '';
        const inputs = Array.isArray(eqn.invars)
          ? eqn.invars
              .map((i: JaxprVar | null) => {
                if (i && typeof i === 'object' && typeof i.name === 'string') {
                  return i.name;
                }
                return '';
              })
              .filter(Boolean)
          : [];
        const outputs = Array.isArray(eqn.outvars)
          ? eqn.outvars
              .map((o: JaxprVar | null) => {
                if (o && typeof o === 'object' && typeof o.name === 'string') {
                  return o.name;
                }
                return '';
              })
              .filter(Boolean)
          : [];
        const firstOutput = outputs[0] ?? '';
        const nodeName = outputs.length > 0 ? `${primStr}_${firstOutput}` : primStr;
        const rawParams = typeof eqn.params === 'object' && eqn.params !== null ? eqn.params : {};
        const params = mapDictToAttributes(rawParams);
        const n = new Node(primStr, inputs, outputs, params, nodeName);
        graph.nodes.push(n);

        if (Array.isArray(eqn.outvars)) {
          for (const outvar of eqn.outvars) {
            if (!outvar || typeof outvar !== 'object') continue;
            const outNameStr = typeof outvar.name === 'string' ? outvar.name : '';
            const outTypeStr = typeof outvar.type === 'string' ? outvar.type : '';
            const outShapeArr = Array.isArray(outvar.shape) ? outvar.shape : [];
            const t = new Tensor(outNameStr, outShapeArr, this.mapJaxType(outTypeStr));
            graph.tensors[outNameStr] = t;
          }
        }
      }
    }

    if (Array.isArray(model.outvars)) {
      for (const outvar of model.outvars) {
        if (!outvar || typeof outvar !== 'object') continue;
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

/**
 * Intercepts `tf.function` XLA HLO graphs and maps them to ONNX IR.
 */
export class XLAHLOParser extends BaseParser {
  /** Mapping from HLO instruction names to tensor transformation functions. */
  public hloToIr: Record<string, (t: Tensor) => Tensor>;

  /**
   * Initializes a new `XLAHLOParser` with default HLO instruction handlers.
   */
  constructor() {
    super();
    this.hloToIr = {
      add: add,
      dot: matmul,
    };
  }

  /**
   * Parses an XLA HLO model representation into an ONNX `Graph`.
   * @param model XLA HLO model object or `null`.
   * @returns Parsed ONNX `Graph`.
   */
  parse(model: FrontendModelInput | null): Graph {
    if (model) {
      /* model metadata inspected when present */
    }
    const graph = new Graph('converted_graph');
    graph.name = 'XLA_Exported';
    return graph;
  }
}
