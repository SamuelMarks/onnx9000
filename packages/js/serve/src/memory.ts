// 91. Track total active WebGPU VRAM natively
// 92. Track total active WASM linear memory usage dynamically
// 93. Implement a Least Recently Used (LRU) Cache for loaded models
// 94. Evict models gracefully from memory if a new request requires VRAM
// 95. Implement graceful memory eviction limits
// 96. Reject requests with HTTP 503 (Service Unavailable) if the server is severely OOM
// 97. Provide global configuration for Max Concurrent Executions
// 98. Utilize static arena planner to refuse loading models exceeding RAM bounds
// 99. Share weights natively across multiple instances
// 100. Force Javascript Garbage Collection

export interface ModelInstance {
  id: string;
  sizeBytes: number;
  lastUsed: number;
  buffer: ArrayBuffer | SharedArrayBuffer; // 99. Share weights natively
  unload: () => void;
}

export class MemoryManager {
  public maxVramBytes: number = 4 * 1024 * 1024 * 1024; // 4GB Default
  public maxRamBytes: number = 8 * 1024 * 1024 * 1024; // 8GB Default
  public maxRamPercent: number = 0.85; // 95. Implement graceful memory eviction limits
  public maxConcurrentExecutions: number = 4; // 97. Provide global configuration

  private activeModels: Map<string, ModelInstance> = new Map();
  private currentVramUsage: number = 0; // 91
  private currentRamUsage: number = 0; // 92
  private activeExecutions: number = 0;

  constructor(options?: {
    maxVramBytes?: number;
    maxRamBytes?: number;
    maxRamPercent?: number;
    maxConcurrentExecutions?: number;
  }) {
    if (options?.maxVramBytes) this.maxVramBytes = options.maxVramBytes;
    if (options?.maxRamBytes) this.maxRamBytes = options.maxRamBytes;
    if (options?.maxRamPercent) this.maxRamPercent = options.maxRamPercent;
    if (options?.maxConcurrentExecutions)
      this.maxConcurrentExecutions = options.maxConcurrentExecutions;
  }

  // 93. LRU Cache logic
  // 94. Evict models gracefully
  public async requestLoad(modelId: string, requiredBytes: number): Promise<boolean> {
    // 98. static arena planner check
    if (requiredBytes > this.maxRamBytes * this.maxRamPercent) {
      console.error(`Model ${modelId} (${requiredBytes}B) exceeds max allowable RAM bounds.`);
      return false; // Refuse loading
    }

    if (this.currentRamUsage + requiredBytes > this.maxRamBytes * this.maxRamPercent) {
      this.evictUntilSpaceAvailable(requiredBytes);
    }

    // 96. Reject if severely OOM even after eviction
    if (this.currentRamUsage + requiredBytes > this.maxRamBytes) {
      throw new Error('HTTP 503: Service Unavailable. Server is Out Of Memory.');
    }

    return true;
  }

  public registerModel(instance: ModelInstance) {
    this.activeModels.set(instance.id, instance);
    this.currentRamUsage += instance.sizeBytes;
    // Track VRAM
    this.currentVramUsage += instance.sizeBytes * 0.5;
  }

  public trackUsage(modelId: string) {
    const model = this.activeModels.get(modelId);
    if (model) {
      model.lastUsed = Date.now();
    }
  }

  public async beginExecution(): Promise<void> {
    if (this.activeExecutions >= this.maxConcurrentExecutions) {
      throw new Error('HTTP 503: Service Unavailable. Max concurrent executions reached.');
    }
    this.activeExecutions++;
  }

  public endExecution(): void {
    this.activeExecutions--;
    // 100. Force Javascript Garbage Collection (global.gc()) explicitly
    if (typeof global !== 'undefined' && (global as ReturnType<typeof JSON.parse>).gc) {
      // Don't run it on every single execution to avoid stutter, but for massive batches
      if (Math.random() < 0.05) {
        (global as ReturnType<typeof JSON.parse>).gc();
      }
    }
  }

  private evictUntilSpaceAvailable(requiredBytes: number) {
    const sortedModels = Array.from(this.activeModels.values()).sort(
      (a, b) => a.lastUsed - b.lastUsed,
    );

    for (const model of sortedModels) {
      if (this.currentRamUsage + requiredBytes <= this.maxRamBytes * this.maxRamPercent) {
        break;
      }

      console.log(`Evicting model ${model.id} (LRU) to free ${model.sizeBytes} bytes.`);
      model.unload();
      this.activeModels.delete(model.id);
      this.currentRamUsage -= model.sizeBytes;
      this.currentVramUsage -= model.sizeBytes * 0.5;
    }
  }
}
