import { Router } from './router';
import { addKServeRoutes } from './kserve';
import { addOpenAIRoutes } from './openai';
import { serveNode } from './node';
import { createLambdaHandler } from './lambda';
import { DynamicBatcher } from './batcher';
import { ModelEnsemble, EnsembleConfig } from './ensemble';
import { MemoryManager } from './memory';
import { ModelRepository } from './repository';
import { KVCacheManager, KVSyncAdapter } from './kv_cache';
import { HashRing, PeerRegistry, proxyRequest } from './routing';
import { addMetricsRoutes, globalMetrics } from './metrics';
import { addDashboardRoutes } from './dashboard';
import { globalLogger, LogLevel } from './logger';
import { applyMiddlewares } from './middleware';
import { runCli } from './cli';
import { createTensorRTSession } from './tensorrt';

export class Onnx9000Server {
  public router: Router;
  public kvCache: KVCacheManager;
  public peerRegistry: PeerRegistry;

  constructor() {
    this.router = new Router();
    this.kvCache = new KVCacheManager();
    this.peerRegistry = new PeerRegistry();
    addKServeRoutes(this, this.router);
    addOpenAIRoutes(this, this.router);
    addMetricsRoutes(this.router);
    addDashboardRoutes(this.router);
  }

  // Generic Edge fetch handler
  public fetch = async (req: Request): Promise<Response> => {
    globalLogger.info(`Incoming request: ${req.method} ${req.url}`);
    const wrappedHandle = applyMiddlewares((r, params) => this.router.handle(r));
    return wrappedHandle(req, {});
  };
}

export function createServer(): Onnx9000Server {
  return new Onnx9000Server();
}

const defaultServer = createServer();

export {
  serveNode,
  createLambdaHandler,
  DynamicBatcher,
  ModelEnsemble,
  MemoryManager,
  ModelRepository,
  KVCacheManager,
  HashRing,
  PeerRegistry,
  proxyRequest,
  globalLogger,
  globalMetrics,
  LogLevel,
  runCli,
  createTensorRTSession,
  type EnsembleConfig,
  type KVSyncAdapter,
};

export default {
  fetch: defaultServer.fetch,
};
