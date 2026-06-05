export type RequestHandler = (
  req: Request,
  params: Record<string, string>,
) => Response | Promise<Response>;

export interface Route {
  method: string;
  pattern: RegExp;
  keys: string[];
  handler: RequestHandler;
}

export class Router {
  private routes: Route[] = [];

  public add(method: string, path: string, handler: RequestHandler) {
    const keys: string[] = [];
    let regexPath = path.replace(/:([^\/]+)/g, (_, key) => {
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

  public get(path: string, handler: RequestHandler) {
    this.add('GET', path, handler);
  }
  public post(path: string, handler: RequestHandler) {
    this.add('POST', path, handler);
  }
  public put(path: string, handler: RequestHandler) {
    this.add('PUT', path, handler);
  }
  public delete(path: string, handler: RequestHandler) {
    this.add('DELETE', path, handler);
  }
  public all(path: string, handler: RequestHandler) {
    this.add('ALL', path, handler);
  }

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
            const key = route.keys[i];
            if (key) {
              params[key] = match[i + 1] || '';
            }
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
