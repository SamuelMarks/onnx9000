import { describe, expect, it } from 'vitest';
import { Simplify } from '../src/index.js';

describe('Simplify', () => {
  it('should eliminate Identity and Dropout nodes and rewire inputs', () => {
    const s = new Simplify();
    expect(s.run()).toBe('[simplify] processed');

    const simplified = s.eliminateRedundantOps([
      { opType: 'Identity', inputs: ['in'], output: 't1' },
      { opType: 'Dropout', inputs: ['t1'], output: 't2' },
      { opType: 'Identity', inputs: [], output: 'empty_id' },
      { opType: 'Relu', inputs: ['t2'], output: 'out' },
    ]);
    expect(simplified).toEqual([
      { opType: 'Identity', inputs: [], output: 'empty_id' },
      { opType: 'Relu', inputs: ['in'], output: 'out' },
    ]);
  });
});
