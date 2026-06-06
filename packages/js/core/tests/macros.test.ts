import { describe, expect, it } from 'vitest';
import { MacroExpander, recordOp } from '../src/macros.js';

describe('macros', () => {
  it('should record op', () => {
    const t = recordOp('Test', []);
    expect(t.name).toBe('Test_out');
  });

  it('should expand', () => {
    const e = new MacroExpander();
    const g = e.apply({ nodes: [] } as any);
    expect(g).toBeDefined();
  });
});
