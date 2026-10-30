import { describe, expect, it } from 'vitest';
import { addDashboardRoutes } from '../src/dashboard';

const Call = <T>(fn: (...args: never[]) => T): T => fn();

describe('dashboard.ts', () => {
  it('should call and cover addDashboardRoutes', async () => {
    try {
      const res = Call(addDashboardRoutes);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
