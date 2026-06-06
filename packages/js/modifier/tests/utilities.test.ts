import { describe, expect, it } from 'vitest';
import { ModifierUtilities } from '../src/components/utilities.js';

describe('ModifierUtilities', () => {
  it('should change batch size', () => {
    const mutator: any = {
      graph: {
        inputs: [{ name: 'in', shape: [1, 2] }],
        outputs: [],
        valueInfo: [],
      },
      overrideShape: function (_n: string, s: any) {
        this.graph.inputs[0].shape = s;
      },
    };
    const utils = new ModifierUtilities(mutator);
    utils.changeBatchSize(8);
    expect(mutator.graph.inputs[0].shape[0]).toBe(8);
  });
});
