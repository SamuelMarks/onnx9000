import { describe, it, expect } from 'vitest';
import { addKServeRoutes } from '../src/kserve.js';
import { Router } from '../src/router.js';

describe('kserve', () => {
  it('should add routes and handle health', async () => {
    const r = new Router();
    addKServeRoutes({} as any, r);

    const req = new Request('http://localhost/v2/health/ready');
    const res = await r.handle(req);
    expect(res.status).toBe(200);
    const json = await res.json();
    expect((json as any).ready).toBe(true);
  });
});
