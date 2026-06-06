// @ts-nocheck
import { describe, expect, it } from 'vitest';
import { PipelineNode } from '../src/core/PipelineNode.js';

describe('PipelineNode', () => {
  it('should create node', () => {
    const node = new PipelineNode(
      { sourceFramework: 'a', targetFramework: 'b', activeFile: '' },
      'desc',
    );
    expect(node.id).toBeDefined();
    expect(node.timestamp).toBeDefined();
  });
});
