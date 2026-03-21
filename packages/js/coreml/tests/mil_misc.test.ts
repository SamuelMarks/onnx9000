import { describe, it, expect } from 'vitest';
import {
  Builder,
  MILDataType,
  TensorType,
  ScalarType,
  TupleType,
  CoreMLExportError,
  UnsupportedOpError,
  ThermalThrottlingWarning,
  ANELimitsExceededWarning,
  DoubleDowncastWarning,
  assertDeterministicBuild,
  Model,
  lintMILProgram,
  Program,
  detectAndMapGenAITopologies,
  MILPrinter,
} from '../src/index.js';
import { Graph, Node as ONNXNode } from '@onnx9000/core';

describe('MIL Types', () => {
  it('covers type hierarchy correctly', () => {
    const t = new TensorType(MILDataType.FLOAT32, [1, 2]);
    expect(t.isTensor()).toBe(true);
    expect(t.isScalar()).toBe(false);
    expect(t.isTuple()).toBe(false);
    expect(t.toString()).toBe('tensor<fp32, [1,2]>');

    const s = new ScalarType(MILDataType.INT32);
    expect(s.isTensor()).toBe(false);
    expect(s.isScalar()).toBe(true);
    expect(s.isTuple()).toBe(false);
    expect(s.toString()).toBe('int32');

    const tu = new TupleType([t, s]);
    expect(tu.isTensor()).toBe(false);
    expect(tu.isScalar()).toBe(false);
    expect(tu.isTuple()).toBe(true);
    expect(tu.toString()).toBe('tuple<tensor<fp32, [1,2]>, int32>');
  });
});

describe('MIL Errors', () => {
  it('covers error instantiation', () => {
    expect(new CoreMLExportError('msg').name).toBe('CoreMLExportError');
    expect(new UnsupportedOpError('FakeOp', 'Because').name).toBe('UnsupportedOpError');
    expect(new UnsupportedOpError('FakeOp').message).toContain('FakeOp');
    expect(new ThermalThrottlingWarning('MatMul').name).toBe('ThermalThrottlingWarning');
    expect(new ANELimitsExceededWarning('Too big').name).toBe('ANELimitsExceededWarning');
    expect(new DoubleDowncastWarning().name).toBe('DoubleDowncastWarning');
  });
});

describe('MIL Builder', () => {
  it('covers missing builder paths', () => {
    const builder = new Builder();
    // Use try/catch since builder throws string errors directly
    expect(() => builder.createBlock('orphan')).toThrow('No active function');
    expect(() => builder.addOp('op', {}, [])).toThrow('No active block');

    const fn = builder.createFunction('test', [], []);
    const block = builder.createBlock('b0');

    // scalar & tuple
    builder.scalar(MILDataType.FLOAT32);
    builder.tuple([]);

    // mul, sub, relu
    const v1 = builder.createVar('v1', builder.tensor(MILDataType.FLOAT32, [1]));
    const v2 = builder.createVar('v2', builder.tensor(MILDataType.FLOAT32, [1]));
    builder.sub(v1, v2);
    builder.mul(v1, v2);
    builder.relu(v1);

    builder.setBlock(block); // coverage for setBlock
  });
});

describe('MIL Deterministic Build', () => {
  it('sorts inputs and outputs and metadata', () => {
    const m: Model = {
      specificationVersion: 1,
      description: {
        input: [{ name: 'z' }, { name: 'a' }],
        output: [{ name: 'y' }, { name: 'b' }],
        metadata: {
          creatorDefined: {
            b_key: '1',
            a_key: '2',
          },
        },
      },
    };
    assertDeterministicBuild(m);
    expect(m.description?.input[0]?.name).toBe('a');
    expect(m.description?.output[0]?.name).toBe('b');
    const keys = Object.keys(m.description?.metadata?.creatorDefined!);
    expect(keys[0]).toBe('a_key');
  });
});

describe('MIL Linter', () => {
  it('detects undocumented ops', () => {
    const builder = new Builder();
    builder.createFunction('f', [], []);
    builder.createBlock('b');
    builder.addOp('fake_apple_op', {}, []);
    const prog = builder.createProgram();
    expect(() => lintMILProgram(prog)).toThrowError(UnsupportedOpError);
  });
});

describe('MIL GenAI', () => {
  it('detects whisper and stable diffusion topologies', () => {
    const b = new Builder();
    const fn = b.createFunction('f', [], []);
    const block = b.createBlock('b');

    const g1 = new Graph('whisper');
    g1.addNode(new ONNXNode('Conv', [], [], {}, 'encoder.conv1'));
    detectAndMapGenAITopologies(g1, block); // logs whisper

    const g2 = new Graph('sd');
    g2.addNode(new ONNXNode('Conv', [], [], {}, 'down_blocks_1'));
    detectAndMapGenAITopologies(g2, block); // logs sd

    const g3 = new Graph('llama');
    g3.addNode(new ONNXNode('MatMul', [], [], {}, 'q_proj'));
    const v1 = b.createVar('present_key', b.tensor(MILDataType.FLOAT32, []));
    b.addOp('concat', {}, [v1]);
    detectAndMapGenAITopologies(g3, block);
  });
});

describe('MIL Printer', () => {
  it('prints an entire program', () => {
    const b = new Builder();
    const inVar = b.createVar('in', b.tensor(MILDataType.FLOAT32, [1]));
    const outVar = b.createVar('out', b.tensor(MILDataType.FLOAT32, [1]));
    const fn = b.createFunction('main', [inVar], [outVar]);
    const block = b.createBlock('b0');
    const v1 = b.createVar('v1', b.tensor(MILDataType.FLOAT32, [1]));
    const v2 = b.createVar('v2', b.tensor(MILDataType.FLOAT32, [1]));

    b.addOp('add', { x: v1, y: v2 }, [v1], { attr1: 'test' });
    b.addOp('concat', { vals: [v1, v2] }, [v1]);
    block.outputs = [v1];

    const printer = new MILPrinter();
    const str = printer.printProgram(b.createProgram());
    expect(str).toContain('func @main');
    expect(str).toContain('add(x=%v1');
  });
});
