import { describe, it, expect } from 'vitest';
import { ZooMLIRLoweringPass } from '../src/passes/zoo_lowering';
import { Graph, Node } from '@onnx9000/core';

describe('ZooMLIRLoweringPass', () => {
  it('should lower MatMul and Add to linalg operations', () => {
    const graph = new Graph('test');
    const n1 = new Node('MatMul', ['a', 'b'], ['c']);
    const n2 = new Node('Add', ['c', 'd'], ['e']);
    const n3 = new Node('UnknownOp', [], []);
    graph.nodes.push(n1, n2, n3);

    const mlirMod = ZooMLIRLoweringPass.lowerToMLIR(graph);
    expect(mlirMod.opcode).toBe('builtin.module');

    const funcOp = mlirMod.regions[0].blocks[0].operations[0];
    expect(funcOp.opcode).toBe('func.func');

    const innerOps = funcOp.regions[0].blocks[0].operations;
    expect(innerOps.length).toBe(3);
    expect(innerOps[0].opcode).toBe('linalg.matmul');
    expect(innerOps[1].opcode).toBe('linalg.add');
    expect(innerOps[2].opcode).toBe('unmapped.UnknownOp');
  });
});
