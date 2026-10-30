/**
 * @fileoverview Unit tests for frontend parsers (PyTorchFXParser, JAXprParser, XLAHLOParser).
 */
import { Tensor } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { JAXprParser, PyTorchFXParser, XLAHLOParser } from '../src/parsers.js';

describe('parsers.ts', () => {
  it('should instantiate and parse PyTorchFXParser models with all attribute types', () => {
    const parser = new PyTorchFXParser();
    expect(parser.atenToIr['aten.add.Tensor']).toBeDefined();
    expect(parser.atenToIr['aten.mm.default']).toBeDefined();

    const dummyTensor = new Tensor('in', [2, 2], 'float32');
    const addOut = parser.atenToIr['aten.add.Tensor']?.(dummyTensor);
    const mmOut = parser.atenToIr['aten.mm.default']?.(dummyTensor);
    expect(addOut?.name).toBe('Add_out');
    expect(mmOut?.name).toBe('MatMul_out');

    expect(parser.parse(null).nodes).toHaveLength(0);
    expect(parser.parse({}).nodes).toHaveLength(0);

    const graph = parser.parse({
      nodes: [
        null,
        {
          target: 'aten.add.Tensor',
          name: 'add_1',
          args: ['x', 'y', 1],
          kwargs: {
            alpha: 1,
            scale: 0.5,
            mode: 'fast',
            flag: true,
            off: false,
            noneVal: null,
          },
        },
        {
          op: 'aten.mm.default',
        },
      ],
    });

    expect(graph.nodes).toHaveLength(2);
    expect(graph.nodes[0]?.opType).toBe('add.Tensor');
    expect(graph.nodes[0]?.inputs).toEqual(['x', 'y']);
    expect(graph.nodes[0]?.attributes.alpha?.value).toBe(1);
    expect(graph.nodes[0]?.attributes.scale?.value).toBe(0.5);
    expect(graph.nodes[0]?.attributes.mode?.value).toBe('fast');
    expect(graph.nodes[0]?.attributes.flag?.value).toBe(1);
    expect(graph.nodes[0]?.attributes.off?.value).toBe(0);
    expect(graph.nodes[0]?.attributes.noneVal?.type).toBe('UNKNOWN');
    expect(graph.nodes[1]?.opType).toBe('mm');
  });

  it('should instantiate and parse JAXprParser models across invars, constvars, eqns, and outvars', () => {
    const parser = new JAXprParser();
    expect(parser.parse(null).nodes).toHaveLength(0);
    expect(parser.parse({}).nodes).toHaveLength(0);

    const graph = parser.parse({
      invars: [
        null,
        { name: 'x', type: 'f32', shape: [2, 3] },
        { name: 'idx', type: 'i32', shape: [2] },
        {},
      ],
      constvars: [null, { name: 'w', type: 'f64', shape: [3, 4] }, {}],
      eqns: [
        null,
        {
          primitive: 'dot_general',
          invars: [null, { name: 'x' }, { name: 'w' }, {}],
          outvars: [null, { name: 'y', type: 'f32', shape: [2, 4] }, {}],
          params: { precision: 'default', axis: 1 },
        },
        {
          primitive: 'nop',
        },
      ],
      outvars: [null, { name: 'y' }, { name: 'missing' }, {}],
    });

    expect(graph.inputs).toHaveLength(3);
    expect(graph.inputs[0]?.dtype).toBe('float32');
    expect(graph.inputs[1]?.dtype).toBe('int32');
    expect(graph.initializers).toContain('w');
    expect(graph.nodes).toHaveLength(2);
    expect(graph.nodes[0]?.name).toBe('dot_general_y');
    expect(graph.nodes[1]?.name).toBe('nop');
    expect(graph.outputs).toHaveLength(2);
    expect(graph.outputs[0]?.name).toBe('y');
  });

  it('should instantiate and parse XLAHLOParser models', () => {
    const parser = new XLAHLOParser();
    expect(parser.hloToIr.add).toBeDefined();
    expect(parser.hloToIr.dot).toBeDefined();

    const g1 = parser.parse(null);
    expect(g1.name).toBe('XLA_Exported');

    const g2 = parser.parse({ nodes: [] });
    expect(g2.name).toBe('XLA_Exported');
  });
});
