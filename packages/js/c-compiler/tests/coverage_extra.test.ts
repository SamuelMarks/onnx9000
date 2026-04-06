import { describe, it, expect } from 'vitest';
import { BaseCodegenVisitor, PythonFamilyCodegen } from '../src/codegen';
import { CGenerator } from '../src/generator';
import { Graph, Node, Tensor } from '@onnx9000/core';

describe('Coverage Extra', () => {
  it('BaseCodegenVisitor', () => {
    class MockVisitor extends BaseCodegenVisitor {
      visitNode(n: Node) {
        return 'node';
      }
    }
    const graph = new Graph('test');
    graph.addNode(new Node('Add', [], []));
    const vis = new MockVisitor();
    expect(vis.visit(graph)).toBe('node');

    const base = new BaseCodegenVisitor();
    expect(() => base.visitNode(new Node('Add', [], []))).toThrow('Not implemented');
  });

  it('PythonFamilyCodegen', () => {
    const p = new PythonFamilyCodegen();
    p.imports.add('test_import');
    const graph = new Graph('test');
    const code = p.visit(graph);
    expect(code).toContain('import test_import');
  });

  it('CGenerator generateSummary', () => {
    const graph = new Graph('test');
    const comp = new CGenerator(graph);
    expect(comp.generateSummary()).toContain('/* Memory Summary */');
  });

  it('CGenerator intermediate inline data', () => {
    const graph = new Graph('test');
    graph.addNode(new Node('Relu', ['x'], ['y']));
    // Make 'x' an intermediate by giving it data
    graph.tensors['x'] = new Tensor(
      'x',
      [2],
      'float32',
      false,
      true,
      new Uint8Array([0, 0, 128, 63, 0, 0, 0, 64]),
    ); // 1.0f, 2.0f
    graph.initializers.push('x');

    // Make 'y' an intermediate but without data
    graph.addNode(new Node('Relu', ['y'], ['z']));

    // Cpp
    const comp = new CGenerator(graph, 'model_', true);
    const code = comp.generateSource();
    expect(code).toContain('1.000000f, 2.000000f');

    // C
    const comp2 = new CGenerator(graph, 'model_', false);
    const code2 = comp2.generateSource();
    expect(code2).toContain('1.000000f, 2.000000f');
  });

  it('CGenerator branch coverage edge cases', () => {
    const graph = new Graph('test');

    // Line 29: shape with non-number or <= 0
    graph.inputs.push({ name: 'in1', shape: [-1, 'unk'] as Object, type: 'float32' });
    graph.addNode(new Node('Relu', ['in1'], ['out1']));

    // Line 34: tensor with size 0
    graph.tensors['empty_tensor'] = new Tensor(
      'empty_tensor',
      [0],
      'float32',
      false,
      true,
      new Uint8Array([]),
    );
    graph.initializers.push('empty_tensor');

    // Line 143: node with empty string input
    graph.addNode(new Node('Relu', [''], ['out2']));

    const comp = new CGenerator(graph);
    const code = comp.generateSource();

    // We just expect it to generate something without crashing
    expect(code).toBeTruthy();
  });
});
