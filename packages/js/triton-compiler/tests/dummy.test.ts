/**
 * @fileoverview Unit tests for Triton compiler kernel generation and UI component.
 */
import { Graph, Node, ValueInfo } from '@onnx9000/core';
import { describe, expect, test } from 'vitest';
import { generateTriton } from '../src/index.js';

describe('triton-compiler', () => {
  test('generates Triton kernel code for an Add graph', () => {
    const g = new Graph('test_triton');
    g.inputs.push(new ValueInfo('A', [128], 'float32'));
    g.inputs.push(new ValueInfo('B', [128], 'float32'));
    g.outputs.push(new ValueInfo('C', [128], 'float32'));
    g.nodes.push(new Node('Add', ['A', 'B'], ['C'], {}, 'add_0'));

    const code = generateTriton(g, { blockM: 64 });
    expect(code).toContain('@triton.jit');
    expect(code).toContain('def test_triton');
  });
});
