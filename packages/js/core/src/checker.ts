export interface CheckerNode {
  op_type: string;
  inputs?: string[];
  outputs?: string[];
  attributes?: Record<string, number | string | number[] | string[] | boolean | object>;
}

export interface CheckerTensor {
  name: string;
  data_type: string;
  shape?: number[];
  is_initializer?: boolean;
  data_location?: string;
  external_data?: { location?: string };
  raw_data?: Uint8Array;
}

export interface CheckerGraph {
  inputs?: CheckerTensor[];
  initializers?: CheckerTensor[];
  nodes?: CheckerNode[];
  outputs?: string[];
}

export interface Model {
  ir_version?: number;
  producer_name?: string;
  opset_import?: { domain: string; version: number }[];
  graph?: CheckerGraph;
}

export interface Model {
  ir_version?: number;
  producer_name?: string;
  opset_import?: { domain: string; version: number }[];
  graph?: CheckerGraph;
}

export class ValidationContext {
  strict: boolean;
  allow_unrecognized_ops: boolean;
  skip_shape_inference: boolean;
  errors: string[];

  constructor(strict = true, allow_unrecognized_ops = false, skip_shape_inference = false) {
    this.strict = strict;
    this.allow_unrecognized_ops = allow_unrecognized_ops;
    this.skip_shape_inference = skip_shape_inference;
    this.errors = [];
  }
}

export class SchemaRegistry {
  schemas: Record<string, Record<string, Record<string, string>>>;
  constructor() {
    this.schemas = {};
    for (let i = 1; i <= 21; i++) {
      this.schemas[`ai.onnx_v${i.toString()}`] = { Conv: { pads: 'ints', strides: 'ints' } };
    }
    for (let i = 1; i <= 4; i++) {
      this.schemas[`ai.onnx.ml_v${i.toString()}`] = { TreeEnsembleClassifier: {} };
    }
  }

  register_custom_schema(
    domain: string,
    opset: number,
    schema_json: Record<string, Record<string, string>>,
  ) {
    this.schemas[`${domain}_v${opset.toString()}`] = schema_json;
  }

  get_schema(op_type: string, opset: number, domain = 'ai.onnx') {
    const key = `${domain}_v${opset.toString()}`;
    if (!this.schemas[key]) throw new Error(`Unsupported opset: ${key as unknown as string}`);
    if (!this.schemas[key][op_type])
      throw new Error(
        `Unsupported op: ${op_type as unknown as string} in ${key as unknown as string}`,
      );
    return this.schemas[key][op_type];
  }
}

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
        ctx.errors.push(`Invalid dim: ${(dim as unknown as number).toString()}`);
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

function _check_op_specific(node: CheckerNode, ctx: ValidationContext) {
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

export async function check_model_async(model: Model, ctx?: ValidationContext) {
  await Promise.resolve();
  return check_model(model, ctx);
}
