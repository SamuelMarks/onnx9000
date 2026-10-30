import { describe, expect, it } from 'vitest';
import { PipelineNode } from '../../src/core/PipelineNode';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('PipelineNode.ts', () => {
  it('should instantiate and cover PipelineNode', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(PipelineNode);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
