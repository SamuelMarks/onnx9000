import { expect, test } from 'vitest';
import { BaseCodegenVisitor, CFamilyCodegen, PythonFamilyCodegen } from '../src/codegen.js';
import { Graph, Node } from '@onnx9000/core';

test('BaseCodegenVisitor', () => {
  class MockVisitor extends BaseCodegenVisitor {
    visitNode(node: Node): string {
      return 'test';
    }
  }
  const v = new MockVisitor();
  expect(v.getVarName()).toBe('v1');
});

test('CFamilyCodegen', () => {
  const g = new Graph();
  g.name = 'test';
  g.nodes.push(new Node('Add', [], []));
  const v = new CFamilyCodegen();
  const code = v.visit(g);
  expect(code).toContain('#include <stddef.h>');
  expect(code).toContain('void forward_test() {');
  expect(code).toContain('Tensor v1 = op_add();');
});

test('PythonFamilyCodegen', () => {
  const g = new Graph();
  g.name = 'test';
  g.nodes.push(new Node('Add', [], []));
  const v = new PythonFamilyCodegen();
  const code = v.visit(g);
  expect(code).toContain('class Model:');
  expect(code).toContain('def forward_test(self):');
  expect(code).toContain('v1 = add()');
});
