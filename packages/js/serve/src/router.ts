/**
 * @fileoverview router.ts
 * Provides router functionality for the serve package.
 */
export type RequestHandler = (
  req: Request,
  params: Record<string, string>,
) => Response | Promise<Response>;

/** Interface `Route` definition. */
export interface Route {
  /** Property `method`. */
  method: string;
  /** Property `pattern`. */
  pattern: RegExp;
  /** Property `keys`. */
  keys: string[];
  /** Property `handler`. */
  handler: RequestHandler;
}

/** Represents the `Router` class. */
export class Router {
  /** Property `routes` of `Router`. */
  private routes: Route[] = [];

  /**
   * Executes `add` on `Router`.
   * @param method Parameter `method`.
   * @param path Parameter `path`.
   * @param handler Parameter `handler`.
   * @returns Result of `add`.
   */
  public add(method: string, path: string, handler: RequestHandler) {
    const keys: string[] = [];
    let regexPath = path.replace(/:([^/]+)/g, (_, key) => {
      keys.push(key);
      return '([^/]+)';
    });
    // Support wildcard
    regexPath = regexPath.replace(/\*/g, '.*');
    this.routes.push({
      method: method.toUpperCase(),
      pattern: new RegExp(`^${regexPath}$`),
      keys,
      handler,
    });
  }

  /**
   * Executes `get` on `Router`.
   * @param path Parameter `path`.
   * @param handler Parameter `handler`.
   * @returns Result of `get`.
   */
  public get(path: string, handler: RequestHandler) {
    this.add('GET', path, handler);
  }
  /**
   * Executes `post` on `Router`.
   * @param path Parameter `path`.
   * @param handler Parameter `handler`.
   * @returns Result of `post`.
   */
  public post(path: string, handler: RequestHandler) {
    this.add('POST', path, handler);
  }
  /**
   * Executes `put` on `Router`.
   * @param path Parameter `path`.
   * @param handler Parameter `handler`.
   * @returns Result of `put`.
   */
  public put(path: string, handler: RequestHandler) {
    this.add('PUT', path, handler);
  }
  /**
   * Executes `delete` on `Router`.
   * @param path Parameter `path`.
   * @param handler Parameter `handler`.
   * @returns Result of `delete`.
   */
  public delete(path: string, handler: RequestHandler) {
    this.add('DELETE', path, handler);
  }
  /**
   * Executes `all` on `Router`.
   * @param path Parameter `path`.
   * @param handler Parameter `handler`.
   * @returns Result of `all`.
   */
  public all(path: string, handler: RequestHandler) {
    this.add('ALL', path, handler);
  }

  /**
   * Executes `handle` on `Router`.
   * @param req Parameter `req`.
   * @returns Result of `handle`.
   */
  public async handle(req: Request): Promise<Response> {
    const url = new URL(req.url);
    const method = req.method.toUpperCase();

    // CORS preflight
    if (method === 'OPTIONS') {
      return new Response(null, {
        status: 204,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type, Authorization, x-kserve-format',
        },
      });
    }

    for (const route of this.routes) {
      if (route.method === method || route.method === 'ALL') {
        const match = url.pathname.match(route.pattern);
        if (match) {
          const params: Record<string, string> = {};
          for (let i = 0; i < route.keys.length; i++) {
            const key = route.keys[i] as string;
            params[key] = match[i + 1] as string;
          }
          try {
            const response = await route.handler(req, params);
            // Append CORS headers
            if (!response.headers.has('Access-Control-Allow-Origin')) {
              response.headers.set('Access-Control-Allow-Origin', '*');
            }
            return response;
          } catch (_err) {
            const err = _err instanceof Error ? _err : new Error(String(_err));
            return new Response(JSON.stringify({ error: err.message }), {
              status: 500,
              headers: {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
              },
            });
          }
        }
      }
    }

    return new Response(JSON.stringify({ error: 'Not Found' }), {
      status: 404,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
      },
    });
  }
}
