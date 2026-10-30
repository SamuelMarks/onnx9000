/**
 * @fileoverview index.ts
 * Provides index functionality for the transformers package.
 */
export class WorkerPipeline {
  /** Property `worker` of `WorkerPipeline`. */
  worker: Worker;

  /**
   * Initializes a new `WorkerPipeline` instance.
   * @param workerPath Parameter `workerPath`.
   */
  constructor(workerPath: string) {
    this.worker = new Worker(workerPath);
  }

  /**
   * Executes `run` on `WorkerPipeline`.
   * @param task Parameter `task`.
   * @param model Parameter `model`.
   * @param input Parameter `input`.
   * @returns Result of `run`.
   */
  async run(
    task: string,
    model: string,
    input: ReturnType<typeof JSON.parse>,
  ): Promise<ReturnType<typeof JSON.parse>> {
    return new Promise((resolve, reject) => {
      const id = Math.random().toString(36).substr(2, 9);
      const handler = (e: MessageEvent) => {
        if (e.data.id === id) {
          this.worker.removeEventListener('message', handler);
          if (e.data.error) reject(new Error(e.data.error));
          else resolve(e.data.result);
        }
      };
      this.worker.addEventListener('message', handler);
      this.worker.postMessage({ id, task, model, input });
    });
  }

  // 242. Zero-Copy transfer
  /**
   * Executes `runZeroCopy` on `WorkerPipeline`.
   * @param task Parameter `task`.
   * @param model Parameter `model`.
   * @param buffer Parameter `buffer`.
   * @returns Result of `runZeroCopy`.
   */
  async runZeroCopy(
    task: string,
    model: string,
    buffer: Float32Array,
  ): Promise<ReturnType<typeof JSON.parse>> {
    return new Promise((resolve, reject) => {
      const id = Math.random().toString(36).substr(2, 9);
      const handler = (e: MessageEvent) => {
        if (e.data.id === id) {
          this.worker.removeEventListener('message', handler);
          if (e.data.error) reject(new Error(e.data.error));
          else resolve(e.data.result);
        }
      };
      this.worker.addEventListener('message', handler);
      this.worker.postMessage({ id, task, model, buffer }, [buffer.buffer]);
    });
  }

  // 244. SharedArrayBuffer
  /**
   * Executes `createSharedMemory` on `WorkerPipeline`.
   * @param size Parameter `size`.
   * @returns Result of `createSharedMemory`.
   */
  createSharedMemory(size: number) {
    return new SharedArrayBuffer(size);
  }
}
