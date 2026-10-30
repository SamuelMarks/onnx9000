/**
 * @fileoverview index.ts
 * Provides index functionality for the serve package.
 */

import { DynamicBatcher } from './batcher';
import { runCli } from './cli';
import { addDashboardRoutes } from './dashboard';
import { type EnsembleConfig, ModelEnsemble } from './ensemble';
import { addKServeRoutes } from './kserve';
import { KVCacheManager, type KVSyncAdapter } from './kv_cache';
import { createLambdaHandler } from './lambda';
import { globalLogger, LogLevel } from './logger';
import { MemoryManager } from './memory';
import { addMetricsRoutes, globalMetrics } from './metrics';
import { applyMiddlewares } from './middleware';
import { serveNode } from './node';
import { addOpenAIRoutes } from './openai';
import { ModelRepository } from './repository';
import { Router } from './router';
import { HashRing, PeerRegistry, proxyRequest } from './routing';
import { createTensorRTSession } from './tensorrt';

/** Represents the `Onnx9000Server` class. */
export class Onnx9000Server {
  /** Property `router` of `Onnx9000Server`. */
  public router: Router;
  /** Property `kvCache` of `Onnx9000Server`. */
  public kvCache: KVCacheManager;
  /** Property `peerRegistry` of `Onnx9000Server`. */
  public peerRegistry: PeerRegistry;

  /** Initializes a new `Onnx9000Server` instance. */
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
  /** Property `fetch` of `Onnx9000Server`. */
  public fetch = async (req: Request): Promise<Response> => {
    globalLogger.info(`Incoming request: ${req.method} ${req.url}`);
    const wrappedHandle = applyMiddlewares((r, _params) => this.router.handle(r));
    return wrappedHandle(req, {});
  };
}

/**
 * Executes `createServer`.
 * @returns Result of `createServer`.
 */
export function createServer(): Onnx9000Server {
  return new Onnx9000Server();
}

const defaultServer = createServer();

export {
  createLambdaHandler,
  createTensorRTSession,
  DynamicBatcher,
  type EnsembleConfig,
  globalLogger,
  globalMetrics,
  HashRing,
  KVCacheManager,
  type KVSyncAdapter,
  LogLevel,
  MemoryManager,
  ModelEnsemble,
  ModelRepository,
  PeerRegistry,
  proxyRequest,
  runCli,
  serveNode,
};

export default {
  fetch: defaultServer.fetch,
};
