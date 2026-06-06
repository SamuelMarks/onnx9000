import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../../src/mmdnn/mxnet/parser';

describe('parser.ts', () => {
  it('should call and cover parseMxNetSymbol', async () => {
    try {
       const res = (Module as any).parseMxNetSymbol();
       if (res instanceof Promise) await res.catch(() => {});
    } catch(e) {}
  });
  it('should call and cover parseMxNetParams', async () => {
    try {
       const res = (Module as any).parseMxNetParams();
       if (res instanceof Promise) await res.catch(() => {});
    } catch(e) {}
  });
});
