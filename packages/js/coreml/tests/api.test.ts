import { describe, it, expect } from 'vitest';
import { convertToCoreML, importCoreML, buildMLPackage } from '../src/api.js';
import { Graph } from '@onnx9000/core';
import { Program } from '../src/mil/ast.js';
import { MLPackageBuilder } from '../src/mlpackage.js';

describe('API Functions', () => {
  it('should convert to CoreML', () => {
    const graph: Graph = {
      name: 'test',
      inputs: [{ name: 'x', dtype: 'float32', shape: [1] }],
      outputs: [{ name: 'y', dtype: 'float32', shape: [1] }],
      initializers: [],
      tensors: {},
      nodes: [{ opType: 'Identity', inputs: ['x'], outputs: ['y'], attributes: {} }],
    };
    const program = convertToCoreML(graph);
    expect(program).toBeDefined();
  });

  it('should import CoreML', () => {
    const program = new Program();
    const graph = importCoreML(program);
    expect(graph).toBeDefined();
  });

  it('should build MLPackage', () => {
    const builder = buildMLPackage({
      description: '',
      version: 1,
      mainProgram: new Program(),
    });
    expect(builder).toBeInstanceOf(MLPackageBuilder);
  });
});
