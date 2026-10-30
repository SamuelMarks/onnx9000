import { Node } from '@onnx9000/core';
import { describe, expect, it } from 'vitest';
import { mapOnnxNodeToTFLite } from '../src/compiler/operators.js';

describe('operators', () => {
  it('should map', () => {
    expect(mapOnnxNodeToTFLite(new Node('Add', [], [], {}))).toBeDefined();
    expect(mapOnnxNodeToTFLite(new Node('Unknown', [], [], {}))).toBeNull();
  });
});
