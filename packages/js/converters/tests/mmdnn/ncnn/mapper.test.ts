import { describe, it, expect } from 'vitest';
import { NcnnMapper } from '../../../src/mmdnn/ncnn/mapper.js';

describe('NcnnMapper', () => {
  it('should map ncnn', () => {
    const param: any = {
      nodes: [{ type: 'Input', name: 'in', bottoms: [], tops: ['out'], attrs: {} }],
    };
    const mapper = new NcnnMapper(param, new ArrayBuffer(0));
    const g = mapper.getGraph();
    expect(g.inputs.length).toBe(1);
  });
});
