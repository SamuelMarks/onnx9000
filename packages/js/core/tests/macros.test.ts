import { describe, expect, it } from 'vitest';
import { Graph } from '../src/ir/graph.js';
import { MacroExpander, recordOp } from '../src/macros.js';

describe('macros', () => {
  it('should record op', () => {
    const t = recordOp('Test', []);
    expect(t.name).toBe('Test_out');
  });

  it('should expand', () => {
    const e = new MacroExpander();
    const g = e.apply(new Graph('test'));
    expect(g).toBeDefined();
  });
});
