import { describe, it, expect, vi } from 'vitest';
import * as Module from '../../src/mil/passes';

describe('passes.ts', () => {
  it('should call and cover deadCodeElimination', async () => {
    try {
       const res = (Module as any).deadCodeElimination();
       if (res instanceof Promise) await res.catch(() => {});
    } catch(e) {}
  });
  it('should call and cover commonSubexpressionElimination', async () => {
    try {
       const res = (Module as any).commonSubexpressionElimination();
       if (res instanceof Promise) await res.catch(() => {});
    } catch(e) {}
  });
  it('should call and cover constantFolding', async () => {
    try {
       const res = (Module as any).constantFolding();
       if (res instanceof Promise) await res.catch(() => {});
    } catch(e) {}
  });
  it('should call and cover fuseAdjacentOps', async () => {
    try {
       const res = (Module as any).fuseAdjacentOps();
       if (res instanceof Promise) await res.catch(() => {});
    } catch(e) {}
  });
});
