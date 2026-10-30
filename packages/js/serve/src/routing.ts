/**
 * @fileoverview routing.ts
 * Provides routing functionality for the serve package.
 */
// 176. Implement a native Serverless Hash-Ring router
export class HashRing {
  /** Property `nodes` of `HashRing`. */
  private nodes: string[] = [];

  /**
   * Executes `addNode` on `HashRing`.
   * @param node Parameter `node`.
   * @returns Result of `addNode`.
   */
  public addNode(node: string) {
    if (!this.nodes.includes(node)) {
      this.nodes.push(node);
      this.nodes.sort();
    }
  }

  /**
   * Executes `removeNode` on `HashRing`.
   * @param node Parameter `node`.
   * @returns Result of `removeNode`.
   */
  public removeNode(node: string) {
    this.nodes = this.nodes.filter((n) => n !== node);
  }

  /**
   * Executes `hash` on `HashRing`.
   * @param key Parameter `key`.
   * @returns Result of `hash`.
   */
  private hash(key: string): number {
    let hash = 0;
    for (let i = 0; i < key.length; i++) {
      hash = (hash << 5) - hash + key.charCodeAt(i);
      hash |= 0;
    }
    return Math.abs(hash);
  }

  /**
   * Executes `getNode` on `HashRing`.
   * @param key Parameter `key`.
   * @returns Result of `getNode`.
   */
  public getNode(key: string): string | null {
    if (this.nodes.length === 0) return null;
    const h = this.hash(key);
    return this.nodes[h % this.nodes.length] as string;
  }
}

// 178. Maintain a global peer-to-peer registry of loaded models
/** Represents the `PeerRegistry` class. */
export class PeerRegistry {
  /** Property `registry` of `PeerRegistry`. */
  public registry: Map<string, Set<string>> = new Map(); // ModelName -> Set of NodeURLs

  /**
   * Executes `register` on `PeerRegistry`.
   * @param model Parameter `model`.
   * @param nodeUrl Parameter `nodeUrl`.
   * @returns Result of `register`.
   */
  public register(model: string, nodeUrl: string) {
    if (!this.registry.has(model)) {
      this.registry.set(model, new Set());
    }
    this.registry.get(model)?.add(nodeUrl);
  }

  // 179. Support generic round-robin load balancing
  /** Property `roundRobinIdx` of `PeerRegistry`. */
  private roundRobinIdx: Map<string, number> = new Map();

  /**
   * Executes `getNextNodeForModel` on `PeerRegistry`.
   * @param model Parameter `model`.
   * @returns Result of `getNextNodeForModel`.
   */
  public getNextNodeForModel(model: string): string | null {
    const nodes = Array.from(this.registry.get(model) || []);
    if (nodes.length === 0) return null;

    const idx = this.roundRobinIdx.get(model) || 0;
    const node = nodes[idx % nodes.length] as string;
    this.roundRobinIdx.set(model, idx + 1);
    return node;
  }
}

// 177. If Node A doesn't have `Model X` in memory, transparently proxy
/**
 * Executes `proxyRequest`.
 * @param req Parameter `req`.
 * @param targetUrl Parameter `targetUrl`.
 * @returns Result of `proxyRequest`.
 */
export async function proxyRequest(req: Request, targetUrl: string): Promise<Response> {
  const headers = new Headers(req.headers);
  // 180. Forward HTTP client IPs perfectly via `X-Forwarded-For`
  const clientIp = req.headers.get('cf-connecting-ip') || req.headers.get('x-forwarded-for') || '';
  if (clientIp) {
    headers.set('X-Forwarded-For', clientIp);
  }

  const init: RequestInit = {
    method: req.method,
    headers,
    redirect: 'manual',
  };

  if (req.method !== 'GET' && req.method !== 'HEAD') {
    init.body = await req.arrayBuffer();
  }

  const newUrl = new URL(req.url);
  const target = new URL(targetUrl);
  newUrl.host = target.host;
  newUrl.protocol = target.protocol;

  return fetch(newUrl.toString(), init);
}
