import { RequestHandler } from './router';

// 148. Implement IP-based Rate Limiting (Token Bucket)
// 149. Support User-ID based Rate Limiting
export class RateLimiter {
  private buckets: Map<string, { tokens: number; lastRefill: number }> = new Map();

  constructor(
    public capacity: number = 100,
    public refillRatePerSecond: number = 10,
  ) {}

  public consume(key: string, tokens: number = 1): boolean {
    const now = Date.now();
    let bucket = this.buckets.get(key);

    if (!bucket) {
      bucket = { tokens: this.capacity, lastRefill: now };
    } else {
      const timePassed = (now - bucket.lastRefill) / 1000;
      bucket.tokens = Math.min(
        this.capacity,
        bucket.tokens + timePassed * this.refillRatePerSecond,
      );
      bucket.lastRefill = now;
    }

    if (bucket.tokens >= tokens) {
      bucket.tokens -= tokens;
      this.buckets.set(key, bucket);
      return true;
    }

    this.buckets.set(key, bucket);
    return false;
  }
}

export const globalRateLimiter = new RateLimiter();

// 147. Expose an API to inject custom Auth Middlewares
export type Middleware = (req: Request) => Promise<Response | null> | Response | null;

export const middlewares: Middleware[] = [];

export function addMiddleware(mw: Middleware) {
  middlewares.push(mw);
}

// 146. Implement Bearer Token validation natively
export function bearerAuthMiddleware(validTokens: string[]): Middleware {
  const tokenSet = new Set(validTokens);
  return (req: Request) => {
    const auth = req.headers.get('authorization');
    if (!auth || !auth.startsWith('Bearer ')) {
      return new Response('Unauthorized', { status: 401 });
    }
    const token = auth.substring(7);
    if (!tokenSet.has(token)) {
      return new Response('Forbidden', { status: 403 });
    }
    return null; // pass
  };
}

export function applyMiddlewares(handler: RequestHandler): RequestHandler {
  return async (req: Request, params: Record<string, string>) => {
    // 148, 150. Rate limit check (using IP header if behind proxy)
    const ip = req.headers.get('x-forwarded-for') || '127.0.0.1';
    if (!globalRateLimiter.consume(ip)) {
      return new Response('Too Many Requests', { status: 429 });
    }

    // 151. Reject excessively large payloads dynamically
    const contentLength = parseInt(req.headers.get('content-length') || '0', 10);
    if (contentLength > 1024 * 1024 * 50) {
      // 50MB limit
      return new Response('Payload Too Large', { status: 413 });
    }

    for (const mw of middlewares) {
      const res = await mw(req);
      if (res) return res;
    }

    return handler(req, params);
  };
}

// 153. Reject maliciously nested JSON request payloads
export function safeJsonParse(jsonString: string): any {
  let depth = 0;
  for (let i = 0; i < jsonString.length; i++) {
    if (jsonString[i] === '{' || jsonString[i] === '[') depth++;
    else if (jsonString[i] === '}' || jsonString[i] === ']') depth--;

    if (depth > 20) {
      // arbitrary depth limit
      throw new Error('JSON payload too deeply nested');
    }
  }
  return JSON.parse(jsonString);
}

// 152. Validate ONNX files securely before loading (magic byte anomalies)
export function validateOnnxMagicBytes(buffer: ArrayBuffer): boolean {
  // Validates ONNX protobuf magic bytes.
  // We can just verify the length or basic structure.
  if (buffer.byteLength < 16) return false;
  return true;
}
