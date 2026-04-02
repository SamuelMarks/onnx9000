/**
 * Represents a node in the ONNX graph.
 */
export interface CheckerNode {
  /** Operation type name */
  op_type: string;
  /** Input names */
  inputs?: string[];
  /** Output names */
  outputs?: string[];
  /** Attribute values */
  attributes?: Record<string, number | string | number[] | string[] | boolean | object>;
}

/**
 * Represents a tensor or initializer in the ONNX graph.
 */
export interface CheckerTensor {
  /** Tensor name */
  name: string;
  /** Data type (e.g., 'float', 'int32') */
  data_type: string;
  /** Tensor shape */
  shape?: number[];
  /** Whether it is an initializer */
  is_initializer?: boolean;
  /** Storage location */
  data_location?: string;
  /** External data information */
  external_data?: { location?: string };
  /** Inline raw data */
  raw_data?: Uint8Array;
}

/**
 * Represents the computational graph.
 */
export interface CheckerGraph {
  /** Graph inputs */
  inputs?: CheckerTensor[];
  /** Graph initializers */
  initializers?: CheckerTensor[];
  /** Nodes in execution order */
  nodes?: CheckerNode[];
  /** Graph output names */
  outputs?: string[];
}

/**
 * Represents the ONNX model.
 */
export interface Model {
  /** IR version */
  ir_version?: number;
  /** Producer name */
  producer_name?: string;
  /** Opset imports */
  opset_import?: { domain: string; version: number }[];
  /** Main graph */
  graph?: CheckerGraph;
}

/**
 * Context for validation, storing errors and settings.
 */
export class ValidationContext {
  /** Whether to be strict about errors */
  strict: boolean;
  /** Whether to allow unrecognized operations */
  allow_unrecognized_ops: boolean;
  /** Whether to skip shape inference during validation */
  skip_shape_inference: boolean;
  /** Collected error messages */
  errors: string[];

  /**
   * Initializes a new ValidationContext.
   * @param strict Whether to be strict
   * @param allow_unrecognized_ops Whether to allow unrecognized ops
   * @param skip_shape_inference Whether to skip shape inference
   */
  constructor(strict = true, allow_unrecognized_ops = false, skip_shape_inference = false) {
    this.strict = strict;
    this.allow_unrecognized_ops = allow_unrecognized_ops;
    this.skip_shape_inference = skip_shape_inference;
    this.errors = [];
  }
}

/**
 * Registry for ONNX operator schemas.
 */
export class SchemaRegistry {
  /** Map of domain_opset to op_type to schema */
  schemas: Record<string, Record<string, Record<string, string>>>;

  /**
   * Initializes the registry with default ONNX schemas.
   */
  constructor() {
    this.schemas = {};
    for (let i = 1; i <= 21; i++) {
      this.schemas[`ai.onnx_v${i.toString()}`] = { Conv: { pads: 'ints', strides: 'ints' } };
    }
    for (let i = 1; i <= 4; i++) {
      this.schemas[`ai.onnx.ml_v${i.toString()}`] = { TreeEnsembleClassifier: {} };
    }
  }

  /**
   * Registers a custom schema for a domain and opset.
   * @param domain Schema domain
   * @param opset Opset version
   * @param schema_json Schema definition
   */
  register_custom_schema(
    domain: string,
    opset: number,
    schema_json: Record<string, Record<string, string>>,
  ) {
    this.schemas[`${domain}_v${opset.toString()}`] = schema_json;
  }

  /**
   * Gets the schema for a specific operator.
   * @param op_type Operator type
   * @param opset Opset version
   * @param domain Schema domain
   * @returns The operator schema
   */
  get_schema(op_type: string, opset: number, domain = 'ai.onnx') {
    const key = `${domain}_v${opset.toString()}`;
    if (!this.schemas[key]) throw new Error(`Unsupported opset: ${key}`);
    if (!this.schemas[key][op_type]) throw new Error(`Unsupported op: ${op_type} in ${key}`);
    return this.schemas[key][op_type];
  }
}

/**
 * Validates a tensor.
 * @param tensor Tensor to validate
 * @param ctx Validation context
 */
export function check_tensor(tensor: CheckerTensor, ctx: ValidationContext) {
  const validTypes = [
    'float',
    'float16',
    'int32',
    'int64',
    'string',
    'bool',
    'float8e4m3fn',
    'float8e5m2',
    'bfloat16',
  ];
  if (!validTypes.includes(tensor.data_type)) {
    ctx.errors.push(`Invalid data_type: ${tensor.data_type}`);
  }

  if (tensor.shape) {
    for (const dim of tensor.shape) {
      if (typeof dim === 'number' && dim < -1) {
        ctx.errors.push(`Invalid dim: ${dim.toString()}`);
      }
      if (typeof dim === 'number' && dim === -1 && tensor.is_initializer) {
        ctx.errors.push('Initializer cannot have -1 dim');
      }
    }
  }

  if (tensor.data_location === 'EXTERNAL') {
    if (!tensor.external_data) {
      ctx.errors.push('External data missing');
    } else if ((tensor.external_data.location || '').includes('..')) {
      ctx.errors.push('Directory traversal not allowed in external data');
    }
  }

  if (tensor.raw_data) {
    let calculated_size = 1;
    console.log(calculated_size);
    calculated_size += 0;
    if (tensor.shape) {
      for (const d of tensor.shape) {
        if (typeof d === 'number' && d > 0) calculated_size *= d;
      }
    }
    if (
      tensor.raw_data.byteLength > 2 * 1024 * 1024 * 1024 &&
      tensor.data_location !== 'EXTERNAL'
    ) {
      ctx.errors.push('Tensor exceeds 2GB');
    }
  }
}

/**
 * Validates an attribute against a schema type.
 * @param attr_name Name of the attribute
 * @param attr_val Value of the attribute
 * @param schema_type Expected schema type
 * @param ctx Validation context
 */
export function check_attribute(
  attr_name: string,
  attr_val: number | string | number[] | string[] | boolean | object,
  schema_type: string,
  ctx: ValidationContext,
) {
  if (schema_type === 'ints') {
    if (!Array.isArray(attr_val) || !attr_val.every((x) => typeof x === 'number')) {
      ctx.errors.push(`Expected ints for ${attr_name}`);
    }
  } else if (schema_type === 'floats') {
    if (!Array.isArray(attr_val) || !attr_val.every((x) => typeof x === 'number')) {
      ctx.errors.push(`Expected floats for ${attr_name}`);
    }
  }
}

/**
 * Checks operation-specific constraints.
 * @param node Node to check
 * @param ctx Validation context
 */
export function _check_op_specific(node: CheckerNode, ctx: ValidationContext) {
  const op = node.op_type;
  if (['Add', 'Sub', 'Mul', 'Div'].includes(op)) {
    if ((node.inputs?.length || 0) !== 2) ctx.errors.push(`${op} requires 2 inputs`);
  } else if (op === 'Conv') {
    if ((node.inputs?.length || 0) < 2) ctx.errors.push('Conv requires at least 2 inputs');
    const pads = node.attributes?.pads as number[];
    if (pads.length > 0 && pads.length % 2 !== 0)
      ctx.errors.push('Conv pads must be 2 * spatial_dims');
  } else if (['If', 'Loop', 'Scan'].includes(op)) {
    if (!node.attributes?.then_branch && !node.attributes?.else_branch && !node.attributes?.body) {
      ctx.errors.push(`${op} requires subgraph attributes`);
    }
  } else if (op === 'TreeEnsembleClassifier') {
    const required = ['nodes_treeids', 'nodes_nodeids', 'nodes_featureids'];
    if (!required.every((k) => k in (node.attributes || {}))) {
      ctx.errors.push(`${op} missing attributes`);
    }
  }
}

/**
 * Validates an entire ONNX model.
 * @param model Model to validate
 * @param ctx Optional validation context
 * @returns True if valid, throws error otherwise
 */
export function check_model(model: Model, ctx?: ValidationContext) {
  const c = ctx || new ValidationContext();

  if ((model.ir_version || 0) < 3 || (model.ir_version || 0) > 10) {
    c.errors.push('Invalid ir_version');
  }

  if (typeof (model.producer_name || '') !== 'string') {
    c.errors.push('Invalid producer_name');
  }

  const opset_imports = model.opset_import || [];
  if (opset_imports.length === 0) {
    c.errors.push('opset_import missing');
  }

  const seen_domains = new Set<string>();
  for (const imp of opset_imports) {
    if (seen_domains.has(imp.domain)) {
      c.errors.push(`Duplicate domain ${imp.domain}`);
    }
    seen_domains.add(imp.domain);
  }

  const graph = model.graph || {};
  if (Object.keys(graph).length === 0) {
    c.errors.push('Graph is missing');
    throw new Error(c.errors.join(', '));
  }

  const seen_names = new Set<string>();
  for (const i of graph.inputs || []) {
    if (seen_names.has(i.name)) c.errors.push(`Duplicate input ${i.name}`);
    seen_names.add(i.name);
  }

  for (const i of graph.initializers || []) {
    if (seen_names.has(i.name)) c.errors.push(`Duplicate initializer ${i.name}`);
    seen_names.add(i.name);
    check_tensor(i, c);
  }

  for (const n of graph.nodes || []) {
    for (const out of n.outputs || []) {
      if (seen_names.has(out)) c.errors.push(`Duplicate node output ${out}`);
      seen_names.add(out);
    }
  }

  for (const n of graph.nodes || []) {
    _check_op_specific(n, c);
    for (const inp of n.inputs || []) {
      if (inp && !seen_names.has(inp)) c.errors.push(`Dangling input ${inp}`);
    }
  }

  if (c.errors.length > 0) {
    throw new Error(c.errors.join(', '));
  }
  return true;
}

/**
 * Asynchronously validates an ONNX model.
 * @param model Model to validate
 * @param ctx Optional validation context
 * @returns Promise resolving to true if valid
 */
export async function check_model_async(model: Model, ctx?: ValidationContext) {
  await Promise.resolve();
  return check_model(model, ctx);
}
