import { describe, it, expect, vi } from 'vitest';
import { Router } from '../src/router';
import { ModelRepository } from '../src/repository';
import { MemoryManager } from '../src/memory';
import { globalLogger, LogLevel } from '../src/logger';
import {
  applyMiddlewares,
  bearerAuthMiddleware,
  addMiddleware,
  safeJsonParse,
  validateOnnxMagicBytes,
  globalRateLimiter,
} from '../src/middleware';

describe('Coverage Router & Middleware & Repository', () => {
  it('Router coverage', async () => {
    const router = new Router();
    router.get('/test/:id', async (req, params) => new Response(params.id));
    router.post('/post', async () => new Response('post'));
    router.put('/put', async () => new Response('put'));
    router.delete('/del', async () => new Response('del'));
    router.all('/all', async () => new Response('all'));

    // Test match
    let req = new Request('http://localhost/test/123');
    let res = await router.handle(req);
    expect(await res.text()).toBe('123');

    // Test OPTIONS (CORS)
    req = new Request('http://localhost/test/123', { method: 'OPTIONS' });
    res = await router.handle(req);
    expect(res.status).toBe(204);

    // Test 404
    req = new Request('http://localhost/notfound');
    res = await router.handle(req);
    expect(res.status).toBe(404);

    // Test throwing handler
    router.get('/error', async () => {
      throw new Error('crash');
    });
    req = new Request('http://localhost/error');
    res = await router.handle(req);
    expect(res.status).toBe(500);
    expect(await res.json()).toEqual({ error: 'crash' });
  });

  it('Middleware coverage', async () => {
    // safeJsonParse
    expect(safeJsonParse('{"a":1}')).toEqual({ a: 1 });
    expect(() =>
      safeJsonParse(
        '{"a":{"b":{"c":{"d":{"e":{"f":{"g":{"h":{"i":{"j":{"k":{"l":{"m":{"n":{"o":{"p":{"q":{"r":{"s":{"t":{"u":1}}}}}}}}}}}}}}}}}}}}}',
      ),
    ).toThrow('too deeply nested');

    // validateOnnxMagicBytes
    expect(validateOnnxMagicBytes(new ArrayBuffer(10))).toBe(false);
    expect(validateOnnxMagicBytes(new ArrayBuffer(20))).toBe(true);

    // bearerAuthMiddleware
    const mw = bearerAuthMiddleware(['token1']);

    let req = new Request('http://localhost');
    let res = mw(req);
    expect((res as Response).status).toBe(401);

    req = new Request('http://localhost', { headers: { authorization: 'Bearer bad' } });
    res = mw(req);
    expect((res as Response).status).toBe(403);

    req = new Request('http://localhost', { headers: { authorization: 'Bearer token1' } });
    res = mw(req);
    expect(res).toBeNull();

    // applyMiddlewares
    addMiddleware(async (req) => {
      if (req.headers.has('x-block')) return new Response('blocked', { status: 403 });
      return null;
    });

    const handler = applyMiddlewares(async () => new Response('ok'));

    // test normal
    req = new Request('http://localhost');
    res = (await handler(req, {})) as Response;
    expect(await res.text()).toBe('ok');

    // test block
    req = new Request('http://localhost', { headers: { 'x-block': '1' } });
    res = (await handler(req, {})) as Response;
    expect(await res.text()).toBe('blocked');

    // test rate limit
    globalRateLimiter.capacity = 0; // immediate failure
    (globalRateLimiter as Object).buckets.clear();
    req = new Request('http://localhost', { headers: { 'x-forwarded-for': '127.0.0.1' } });
    res = (await handler(req, {})) as Response;
    expect(res.status).toBe(429);

    globalRateLimiter.capacity = 100; // reset
    (globalRateLimiter as Object).buckets.clear();

    // test payload too large
    req = new Request('http://localhost', { headers: { 'content-length': '100000000' } });
    res = (await handler(req, {})) as Response;
    expect(res.status).toBe(413);
  });

  it('Repository coverage', async () => {
    const mm = new MemoryManager({ maxRamBytes: 1000000 });
    const repo = new ModelRepository(mm);

    const mockFs = {
      watch: vi.fn(),
      existsSync: vi.fn(),
      readFileSync: vi.fn(),
    };
    const mockPath = {
      sep: '/',
      join: (...args: string[]) => args.join('/'),
    };

    await repo.watch(mockFs, mockPath);
    expect(mockFs.watch).toHaveBeenCalled();

    // Invoke watcher callback
    const watcherCb = mockFs.watch.mock.calls[0][2];
    await watcherCb('change', 'model1/1/model.onnx');
    await watcherCb('change', 'ignored.txt');
    await watcherCb('change', 'short/config.json');

    // Test reloadModel when exists
    mockFs.existsSync.mockImplementation((p: string) => {
      if (p.endsWith('config.json')) return true;
      if (p.endsWith('model.onnx')) return true;
      if (p.endsWith('model.safetensors')) return false;
      return true; // base path
    });
    mockFs.readFileSync.mockReturnValue('{"test": 1}');

    await repo.reloadModel('model1', '1', mockFs, mockPath);
    expect(repo.models.get('model1')![0].config.test).toBe(1);

    // Safetensors version
    mockFs.existsSync.mockImplementation((p: string) => {
      if (p.endsWith('config.json')) return false;
      if (p.endsWith('model.onnx')) return false;
      if (p.endsWith('model.safetensors')) return true;
      return true; // base path
    });
    await repo.reloadModel('model2', '1', mockFs, mockPath);
    expect(repo.models.get('model2')![0].platform).toBe('safetensors');

    // Test reloadModel when NOT exists (evict)
    mockFs.existsSync.mockReturnValue(false);
    await repo.reloadModel('model1', '1', mockFs, mockPath);
    expect(repo.models.get('model1')?.length).toBe(0);

    // Evict a non-existent model (coverage branch)
    await repo.reloadModel('model_none', '1', mockFs, mockPath);

    // Watcher without fs
    await repo.watch(null, mockPath);
  });

  it('Logger coverage', () => {
    globalLogger.level = LogLevel.DEBUG;
    globalLogger.debug('debug');
    globalLogger.info('info');
    globalLogger.warn('warn');
    globalLogger.error('error');

    globalLogger.level = LogLevel.ERROR;
    globalLogger.debug('hidden');

    globalLogger.format = 'json';
    globalLogger.error('json error');
    globalLogger.format = 'text'; // reset
  });
});
