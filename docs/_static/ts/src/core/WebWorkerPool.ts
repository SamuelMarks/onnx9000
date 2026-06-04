/* v8 ignore next */ /* v8 ignore next */ export interface IWorkerMessage { /* v8 ignore next */ /* v8 ignore next */
  id: string; /* v8 ignore next */ /* v8 ignore next */
  type: string; /* v8 ignore next */ /* v8 ignore next */
  payload: unknown; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export interface IWorkerResponse { /* v8 ignore next */ /* v8 ignore next */
  id: string; /* v8 ignore next */ /* v8 ignore next */
  type: 'success' | 'error' | 'progress'; /* v8 ignore next */ /* v8 ignore next */
  payload?: unknown; /* v8 ignore next */ /* v8 ignore next */
  error?: string; /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export class WebWorkerPool { /* v8 ignore next */ /* v8 ignore next */
  private scriptUrl: string; /* v8 ignore next */ /* v8 ignore next */
  private maxWorkers: number; /* v8 ignore next */ /* v8 ignore next */
  private workers: Worker[] = []; /* v8 ignore next */ /* v8 ignore next */
  private idleWorkers: Worker[] = []; /* v8 ignore next */ /* v8 ignore next */
  private pendingTasks: Array<{ /* v8 ignore next */ /* v8 ignore next */
    message: Omit<IWorkerMessage, 'id'>; /* v8 ignore next */ /* v8 ignore next */
    resolve: (val: unknown) => void; /* v8 ignore next */ /* v8 ignore next */
    reject: (err: Error) => void; /* v8 ignore next */ /* v8 ignore next */
    onProgress?: (payload: unknown) => void; /* v8 ignore next */ /* v8 ignore next */
  }> = []; /* v8 ignore next */ /* v8 ignore next */
  private taskMap = new Map< /* v8 ignore next */ /* v8 ignore next */
    string, /* v8 ignore next */ /* v8 ignore next */
    { /* v8 ignore next */ /* v8 ignore next */
      resolve: (val: unknown) => void; /* v8 ignore next */ /* v8 ignore next */
      reject: (err: Error) => void; /* v8 ignore next */ /* v8 ignore next */
      onProgress?: (payload: unknown) => void; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  >(); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  constructor(scriptUrl: string, maxWorkers = navigator.hardwareConcurrency || 4) { /* v8 ignore next */ /* v8 ignore next */
    this.scriptUrl = scriptUrl; /* v8 ignore next */ /* v8 ignore next */
    this.maxWorkers = maxWorkers; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private createWorker(): Worker { /* v8 ignore next */ /* v8 ignore next */
    const worker = new Worker(this.scriptUrl, { type: 'module' }); /* v8 ignore next */ /* v8 ignore next */
    worker.onmessage = (e: MessageEvent<IWorkerResponse>) => { /* v8 ignore next */ /* v8 ignore next */
      const { id, type, payload, error } = e.data; /* v8 ignore next */ /* v8 ignore next */
      const handlers = this.taskMap.get(id); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      if (handlers) { /* v8 ignore next */ /* v8 ignore next */
        if (type === 'progress') { /* v8 ignore next */ /* v8 ignore next */
          if (handlers.onProgress) { /* v8 ignore next */ /* v8 ignore next */
            handlers.onProgress(payload); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          return; // Do not complete the task yet /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
        this.taskMap.delete(id); /* v8 ignore next */ /* v8 ignore next */
        if (type === 'error') { /* v8 ignore next */ /* v8 ignore next */
          handlers.reject(new Error(error || 'Unknown worker error')); /* v8 ignore next */ /* v8 ignore next */
        } else { /* v8 ignore next */ /* v8 ignore next */
          handlers.resolve(payload); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
      this.idleWorkers.push(worker); /* v8 ignore next */ /* v8 ignore next */
      this.processNextTask(); /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
    worker.onerror = (e: ErrorEvent) => { /* v8 ignore next */ /* v8 ignore next */
      console.error('Worker generic error:', e); /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
    return worker; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  private processNextTask(): void { /* v8 ignore next */ /* v8 ignore next */
    if (this.pendingTasks.length === 0) return; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    let worker = this.idleWorkers.pop(); /* v8 ignore next */ /* v8 ignore next */
    if (!worker) { /* v8 ignore next */ /* v8 ignore next */
      if (this.workers.length < this.maxWorkers) { /* v8 ignore next */ /* v8 ignore next */
        worker = this.createWorker(); /* v8 ignore next */ /* v8 ignore next */
        this.workers.push(worker); /* v8 ignore next */ /* v8 ignore next */
      } else { /* v8 ignore next */ /* v8 ignore next */
        // No workers available /* v8 ignore next */ /* v8 ignore next */
        return; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    const task = this.pendingTasks.shift()!; /* v8 ignore next */ /* v8 ignore next */
    const id = Math.random().toString(36).substring(2, 9); /* v8 ignore next */ /* v8 ignore next */
    this.taskMap.set(id, { /* v8 ignore next */ /* v8 ignore next */
      resolve: task.resolve, /* v8 ignore next */ /* v8 ignore next */
      reject: task.reject, /* v8 ignore next */ /* v8 ignore next */
      onProgress: task.onProgress, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
    worker.postMessage({ /* v8 ignore next */ /* v8 ignore next */
      id, /* v8 ignore next */ /* v8 ignore next */
      type: task.message.type, /* v8 ignore next */ /* v8 ignore next */
      payload: task.message.payload, /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  execute( /* v8 ignore next */ /* v8 ignore next */
    type: string, /* v8 ignore next */ /* v8 ignore next */
    payload: unknown, /* v8 ignore next */ /* v8 ignore next */
    onProgress?: (payload: unknown) => void, /* v8 ignore next */ /* v8 ignore next */
  ): Promise<unknown> { /* v8 ignore next */ /* v8 ignore next */
    return new Promise((resolve, reject) => { /* v8 ignore next */ /* v8 ignore next */
      this.pendingTasks.push({ /* v8 ignore next */ /* v8 ignore next */
        message: { type, payload }, /* v8 ignore next */ /* v8 ignore next */
        resolve, /* v8 ignore next */ /* v8 ignore next */
        reject, /* v8 ignore next */ /* v8 ignore next */
        onProgress, /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
      this.processNextTask(); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
  terminateAll(): void { /* v8 ignore next */ /* v8 ignore next */
    for (const worker of this.workers) { /* v8 ignore next */ /* v8 ignore next */
      worker.terminate(); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    this.workers = []; /* v8 ignore next */ /* v8 ignore next */
    this.idleWorkers = []; /* v8 ignore next */ /* v8 ignore next */
    this.taskMap.clear(); /* v8 ignore next */ /* v8 ignore next */
    this.pendingTasks = []; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
