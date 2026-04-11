/* eslint-disable */
export class WorkerPipeline {
  worker: Worker;

  constructor(workerPath: string) {
    this.worker = new Worker(workerPath);
  }

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
  createSharedMemory(size: number) {
    return new SharedArrayBuffer(size);
  }
}
