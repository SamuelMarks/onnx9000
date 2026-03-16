import { describe, it, expect } from 'vitest';
import { Graph, Node } from '@onnx9000/core';
import * as c from '../src/index.js';

describe('Compiler Module', () => {
  const g = new Graph('test');
  g.addNode(new Node('Add', ['A', 'B'], ['C']));

  const emptyGraph = new Graph('empty');

  it('should compile to CoreML', () => {
    const mlmodel = c.compileToCoreML(g);
    expect(mlmodel.length).toBe(3);
    expect(() => c.compileToCoreML(emptyGraph)).toThrow('Graph is empty');
  });

  it('should compile to IREE', () => {
    const iree = c.compileToIREE(g);
    expect(iree.length).toBe(4);
    expect(() => c.compileToIREE(emptyGraph)).toThrow('Graph is empty');
  });

  it('should emit WGSL', () => {
    const wgsl = c.emitWGSL(g);
    expect(wgsl).toContain('fn main');
    expect(() => c.emitWGSL(emptyGraph)).toThrow('Graph is empty');
  });

  it('should emit and interpret WVM', () => {
    const wvm = c.emitWVM(g);
    expect(wvm.length).toBe(3);
    expect(() => c.emitWVM(emptyGraph)).toThrow('Graph is empty');

    const interp = new c.WVMInterpreter(wvm);
    expect(interp.execute()).toBe(true);

    const badInterp = new c.WVMInterpreter(new Uint8Array([]));
    expect(() => badInterp.execute()).toThrow('Empty bytecode');
  });
});
