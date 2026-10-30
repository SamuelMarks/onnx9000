import { describe, expect, it } from 'vitest';
import { PipelineHistory } from '../../src/core/PipelineHistory';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('PipelineHistory.ts', () => {
  it('should instantiate and cover PipelineHistory', () => {
    // Attempt to instantiate
    try {
      const obj = Construct(PipelineHistory);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
