import { globalEventBus } from './EventBus';
import { LogLevel } from './Logger';

export interface WorkerMessage {
  id: string;
  type: string;
  payload?: any;
  error?: string;
}

/**
 * Manages pools of Web Workers, facilitating async Promise-based request/responses
 * while handling strict timeouts and graceful termination.
 */
export class WorkerManager {
  private static instance: WorkerManager;
  private worker: Worker | null = null;
  private pendingRequests: Map<
    string,
    { resolve: (val: any) => void; reject: (err: any) => void; timeout: any }
  > = new Map();

  private constructor() {}

  public static getInstance(): WorkerManager {
    if (!WorkerManager.instance) {
      WorkerManager.instance = new WorkerManager();
    }
    return WorkerManager.instance;
  }

  /**
   * Initializes the shared Web Worker instance.
   */
  public initWorker(workerScriptUrl: string = '/worker.js'): void {
    if (this.worker) {
      this.terminate();
    }

    // Feature detect SAB if needed here.
    const supportsSAB = typeof SharedArrayBuffer !== 'undefined';
    console.log(`Worker initialized. SAB Support: ${supportsSAB}`);

    this.worker = new Worker(workerScriptUrl, { type: 'module' });

    this.worker.onmessage = (e: MessageEvent) => {
      const msg: WorkerMessage = e.data;
      if (this.pendingRequests.has(msg.id)) {
        const req = this.pendingRequests.get(msg.id)!;
        clearTimeout(req.timeout);

        if (msg.error) {
          req.reject(new Error(msg.error));
        } else {
          req.resolve(msg.payload);
        }

        this.pendingRequests.delete(msg.id);
      } else {
        // Unhandled messages, perhaps streams to console
        if (msg.type === 'STREAM_STDOUT') {
          globalEventBus.emit('CONSOLE_LOG', {
            level: LogLevel.INFO,
            message: msg.payload,
            timestamp: new Date()
          });
        }
      }
    };

    this.worker.onerror = (err) => {
      console.error('WebWorker crashed unexpectedly', err);
      // Reject all pending
      this.pendingRequests.forEach((req) => req.reject(new Error('Worker crashed')));
      this.pendingRequests.clear();
    };
  }

  /**
   * Dispatches a job to the worker and returns a Promise.
   *
   * @param type - The action type.
   * @param payload - The data.
   * @param timeoutMs - Maximum execution time before aborting.
   */
  public execute(type: string, payload: any, timeoutMs: number = 30000): Promise<any> {
    return new Promise((resolve, reject) => {
      if (!this.worker) {
        return reject(new Error('Worker not initialized.'));
      }

      const id =
        typeof crypto !== 'undefined' && crypto.randomUUID
          ? crypto.randomUUID()
          : Math.random().toString(36).substring(2);

      const timeout = setTimeout(() => {
        if (this.pendingRequests.has(id)) {
          this.pendingRequests.delete(id);
          this.terminate(); // Force kill runaway worker
          this.initWorker(); // Restart a fresh one
          reject(new Error(`Worker task timed out after ${timeoutMs}ms`));
        }
      }, timeoutMs);

      this.pendingRequests.set(id, { resolve, reject, timeout });

      this.worker.postMessage({ id, type, payload });
    });
  }

  /**
   * Terminate the current worker and clear requests.
   */
  public terminate(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }

    this.pendingRequests.forEach((req) => {
      clearTimeout(req.timeout);
      req.reject(new Error('Worker forcefully terminated'));
    });
    this.pendingRequests.clear();
  }
}
