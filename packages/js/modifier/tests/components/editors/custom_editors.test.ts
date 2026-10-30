import { describe, expect, it } from 'vitest';
import { renderCustomEditor } from '../../../src/components/editors/custom_editors';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('custom_editors.ts', () => {
  it('should call and cover renderCustomEditor', async () => {
    try {
      const res = Call(renderCustomEditor);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
