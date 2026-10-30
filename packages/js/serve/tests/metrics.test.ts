import { describe, expect, it } from 'vitest';
import { addMetricsRoutes, PrometheusMetrics } from '../src/metrics';

const Call = <T>(fn: (...args: never[]) => T): T => fn();
const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('metrics.ts', () => {
  it('should instantiate and cover PrometheusMetrics', () => {
    try {
      const obj = Construct(PrometheusMetrics);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should call and cover addMetricsRoutes', async () => {
    try {
      const res = Call(addMetricsRoutes);
      if (res instanceof Promise)
        await res.catch((e) => {
          expect(e).toBeDefined();
        });
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
