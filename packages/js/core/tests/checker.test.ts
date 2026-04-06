import { describe, it, expect } from 'vitest';

import {
  ValidationContext,
  SchemaRegistry,
  check_tensor,
  check_attribute,
  check_model,
  check_model_async,
  CheckerTensor,
  CheckerNode,
  CheckerGraph,
  Model,
} from '../src/checker';

describe('ONNX Checker', () => {
  it('ValidationContext initializes correctly', () => {
    const ctx = new ValidationContext();
    expect(ctx.strict).toBe(true);
    expect(ctx.errors).toEqual([]);
  });

  it('SchemaRegistry manages schemas', () => {
    const registry = new SchemaRegistry();
    expect(registry.get_schema('Conv', 21)).toEqual({ pads: 'ints', strides: 'ints' });

    expect(() => registry.get_schema('Conv', 99)).toThrow('Unsupported opset: ai.onnx_v99');
    expect(() => registry.get_schema('UnknownOp', 21)).toThrow(
      'Unsupported op: UnknownOp in ai.onnx_v21',
    );

    registry.register_custom_schema('custom', 1, { MyOp: {} });
    expect(registry.get_schema('MyOp', 1, 'custom')).toEqual({});
  });

  it('check_tensor handles various tensor configurations', () => {
    const ctx = new ValidationContext();

    // Valid tensor
    const t1: CheckerTensor = { name: 't1', data_type: 'float', shape: [1, 2, 3] };
    check_tensor(t1, ctx);
    expect(ctx.errors.length).toBe(0);

    // Invalid dtype
    const t2: CheckerTensor = { name: 't2', data_type: 'invalid', shape: [1] };
    check_tensor(t2, ctx);
    expect(ctx.errors).toContain('Invalid data_type: invalid');
    ctx.errors = [];

    // Invalid dim
    const t3: CheckerTensor = { name: 't3', data_type: 'float', shape: [-2] };
    check_tensor(t3, ctx);
    expect(ctx.errors).toContain('Invalid dim: -2');
    ctx.errors = [];

    // Initializer with -1
    const t4: CheckerTensor = { name: 't4', data_type: 'float', shape: [-1], is_initializer: true };
    check_tensor(t4, ctx);
    expect(ctx.errors).toContain('Initializer cannot have -1 dim');
    ctx.errors = [];

    // External data missing
    const t5: CheckerTensor = {
      name: 't5',
      data_type: 'float',
      shape: [1],
      data_location: 'EXTERNAL',
    };
    check_tensor(t5, ctx);
    expect(ctx.errors).toContain('External data missing');
    ctx.errors = [];

    // Directory traversal
    const t6: CheckerTensor = {
      name: 't6',
      data_type: 'float',
      shape: [1],
      data_location: 'EXTERNAL',
      external_data: { location: '../test' },
    };
    check_tensor(t6, ctx);
    expect(ctx.errors).toContain('Directory traversal not allowed in external data');
    ctx.errors = [];

    // Exceeds 2GB
    const t7: CheckerTensor = {
      name: 't7',
      data_type: 'float',
      shape: [1024, 1024, 1024],
      raw_data: new Uint8Array(2 * 1024 * 1024 * 1024 + 1),
    };
    check_tensor(t7, ctx);
    expect(ctx.errors).toContain('Tensor exceeds 2GB');
    ctx.errors = [];
  });

  it('check_attribute validates attributes correctly', () => {
    const ctx = new ValidationContext();
    check_attribute('pads', [1, 1], 'ints', ctx);
    expect(ctx.errors.length).toBe(0);

    check_attribute('pads', ['1'], 'ints', ctx);
    expect(ctx.errors).toContain('Expected ints for pads');
    ctx.errors = [];

    check_attribute('scales', [1.0, 2.0], 'floats', ctx);
    expect(ctx.errors.length).toBe(0);

    check_attribute('scales', ['1.0'], 'floats', ctx);
    expect(ctx.errors).toContain('Expected floats for scales');
    ctx.errors = [];
  });

  it('check_model handles valid and invalid models', () => {
    // Valid model
    const n1: CheckerNode = { op_type: 'Add', inputs: ['x', 'y'], outputs: ['z'] };
    const i1: CheckerTensor = { name: 'y', data_type: 'float', shape: [1], is_initializer: true };
    const g1: CheckerGraph = {
      inputs: [{ name: 'x', data_type: 'float', shape: [1], is_initializer: false }],
      initializers: [i1],
      nodes: [n1],
    };

    const m1: Model = {
      ir_version: 8,
      producer_name: 'test',
      opset_import: [{ domain: 'ai.onnx', version: 15 }],
      graph: g1,
    };

    expect(check_model(m1)).toBe(true);

    // Invalid models
    const m2: Model = { ir_version: 1 };
    expect(() => check_model(m2)).toThrow(/Invalid ir_version/);
  });

  it('check_model_async runs correctly', async () => {
    const m1: Model = {
      ir_version: 8,
      producer_name: 'test',
      opset_import: [{ domain: 'ai.onnx', version: 15 }],
      graph: {
        inputs: [{ name: 'x', data_type: 'float', shape: [1] }],
        initializers: [{ name: 'y', data_type: 'float', shape: [1], is_initializer: true }],
        nodes: [{ op_type: 'Add', inputs: ['x', 'y'], outputs: ['z'] }],
      },
    };
    const res = await check_model_async(m1);
    expect(res).toBe(true);
  });
});

it('check_model handles missing fields and duplicates', () => {
  // missing opset_import
  const m1: Model = { producer_name: '' as Object };
  expect(() => check_model(m1)).toThrow(/opset_import missing/);

  // duplicate opset_import domain
  const m2: Model = {
    opset_import: [
      { domain: 'ai.onnx', version: 15 },
      { domain: 'ai.onnx', version: 16 },
    ],
    graph: {},
  };
  expect(() => check_model(m2)).toThrow(/Duplicate domain ai.onnx/);

  // duplicate input
  const i: CheckerTensor = { name: 'dup', data_type: 'float', shape: [1] };
  const m3: Model = {
    opset_import: [{ domain: 'ai.onnx', version: 15 }],
    graph: { inputs: [i, i] },
  };
  expect(() => check_model(m3)).toThrow(/Duplicate input dup/);

  // duplicate initializer
  const m4: Model = {
    opset_import: [{ domain: 'ai.onnx', version: 15 }],
    graph: { initializers: [i, i] },
  };
  expect(() => check_model(m4)).toThrow(/Duplicate initializer dup/);

  // duplicate output
  const n: CheckerNode = { op_type: 'Add', inputs: ['x', 'y'], outputs: ['z', 'z'] };
  const m5: Model = { opset_import: [{ domain: 'ai.onnx', version: 15 }], graph: { nodes: [n] } };
  expect(() => check_model(m5)).toThrow(/Duplicate node output z/);

  // dangling input
  const n2: CheckerNode = { op_type: 'Add', inputs: ['x', 'missing'], outputs: ['z'] };
  const m6: Model = { opset_import: [{ domain: 'ai.onnx', version: 15 }], graph: { nodes: [n2] } };
  expect(() => check_model(m6)).toThrow(/Dangling input x/);

  // Invalid producer name
  const m7: Model = { producer_name: 123 as Object, opset_import: [], graph: {} };
  expect(() => check_model(m7)).toThrow(/Invalid producer_name/);
});

it('check_op_specific handles specific node logic', () => {
  const ctx = new ValidationContext();

  // Add
  const n1: CheckerNode = { op_type: 'Add', inputs: ['x'], outputs: ['z'] };
  // We can't easily call _check_op_specific directly if it's not exported. Let's just pass it in a model graph
  const m1: Model = {
    opset_import: [{ domain: 'ai.onnx', version: 15 }],
    graph: { inputs: [], nodes: [n1] },
  };

  // Conv
  const n2: CheckerNode = { op_type: 'Conv', inputs: ['x'], outputs: ['z'] };
  const m2: Model = {
    opset_import: [{ domain: 'ai.onnx', version: 15 }],
    graph: { inputs: [], nodes: [n2] },
  };

  const n3: CheckerNode = {
    op_type: 'Conv',
    inputs: ['x', 'w'],
    outputs: ['z'],
    attributes: { pads: [1, 2, 3] },
  };
  const m3: Model = {
    opset_import: [{ domain: 'ai.onnx', version: 15 }],
    graph: { inputs: [], nodes: [n3] },
  };
  expect(() => check_model(m3)).toThrow(/Conv pads must be 2 \* spatial_dims/);

  // Control flow
  const n4: CheckerNode = { op_type: 'If', inputs: ['x'], outputs: ['z'] };
  const m4: Model = {
    opset_import: [{ domain: 'ai.onnx', version: 15 }],
    graph: { inputs: [], nodes: [n4] },
  };

  // TreeEnsemble
  const n5: CheckerNode = {
    op_type: 'TreeEnsembleClassifier',
    inputs: ['x'],
    outputs: ['z'],
    attributes: { nodes_treeids: [1] },
  };
  const m5: Model = {
    opset_import: [{ domain: 'ai.onnx', version: 15 }],
    graph: { inputs: [], nodes: [n5] },
  };
  expect(() => check_model(m5)).toThrow(/TreeEnsembleClassifier missing attributes/);
});
