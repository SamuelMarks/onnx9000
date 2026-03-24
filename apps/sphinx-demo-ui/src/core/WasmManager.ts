import { globalEventBus } from './EventBus';

export enum WasmState {
  IDLE = 'IDLE',
  LOADING = 'LOADING',
  LOADED = 'LOADED',
  ERROR = 'ERROR'
}

/**
 * Singleton manager responsible for lazily fetching, tracking progress,
 * and instantiating heavy WASM binaries.
 */
export class WasmManager {
  private static instance: WasmManager;
  private _state: WasmState = WasmState.IDLE;
  private _progress: number = 0; // 0 to 100
  private _error: string | null = null;
  /* private _module: WebAssembly.Module | null = null; */
  private _instance: WebAssembly.Instance | null = null;

  private constructor() {}

  /**
   * Retrieves the global singleton instance of the WasmManager.
   */
  public static getInstance(): WasmManager {
    if (!WasmManager.instance) {
      WasmManager.instance = new WasmManager();
    }
    return WasmManager.instance;
  }

  /**
   * Gets the current state of the WASM loading process.
   */
  public get state(): WasmState {
    return this._state;
  }

  /**
   * Gets the current load progress percentage (0-100).
   */
  public get progress(): number {
    return this._progress;
  }

  /**
   * Gets any active error message if state is ERROR.
   */
  public get error(): string | null {
    return this._error;
  }

  /**
   * Resets the manager state to IDLE. Useful for testing.
   */
  public reset(): void {
    this._state = WasmState.IDLE;
    this._progress = 0;
    this._error = null;
    /* this._module = null; */
    this._instance = null;
    globalEventBus.emit('WASM_STATE_CHANGED', this._state);
  }

  /**
   * Initiates the fetch and initialization of the WASM binary.
   * Tracks progress using the `ReadableStream` API.
   *
   * @param url - The URL to the WASM file.
   */
  public async load(url: string = '/onnx9000.wasm'): Promise<void> {
    if (this._state === WasmState.LOADING || this._state === WasmState.LOADED) {
      return;
    }

    this._state = WasmState.LOADING;
    this._progress = 0;
    this._error = null;
    globalEventBus.emit('WASM_STATE_CHANGED', this._state);
    globalEventBus.emit('WASM_PROGRESS', this._progress);

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const contentLength = response.headers.get('content-length');
      const total = contentLength ? parseInt(contentLength, 10) : 0;
      let loaded = 0;

      const body = response.body;
      if (!body) {
        throw new Error('ReadableStream not supported by browser.');
      }

      const reader = body.getReader();
      const chunks: Uint8Array[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }

        if (value) {
          chunks.push(value);
          loaded += value.length;

          if (total > 0) {
            this._progress = Math.round((loaded / total) * 100);
          } else {
            // Fake progress if no content-length
            this._progress = Math.min(99, this._progress + 5);
          }
          globalEventBus.emit('WASM_PROGRESS', this._progress);
        }
      }

      // Concatenate chunks
      const wasmBinary = new Uint8Array(loaded);
      let offset = 0;
      for (const chunk of chunks) {
        wasmBinary.set(chunk, offset);
        offset += chunk.length;
      }

      // Instantiate WASM
      const module = await WebAssembly.compile(wasmBinary.buffer);
      /* this._module = module; */

      const instance = await WebAssembly.instantiate(module, {
        env: {
          abort: () => console.error('WASM abort')
        }
      });
      this._instance = instance;

      this._progress = 100;
      this._state = WasmState.LOADED;
      globalEventBus.emit('WASM_PROGRESS', this._progress);
      globalEventBus.emit('WASM_STATE_CHANGED', this._state);
      globalEventBus.emit('WASM_LOADED', this._instance);
    } catch (err: any) {
      this._error = err.message || 'Failed to load WASM binary';
      this._state = WasmState.ERROR;
      globalEventBus.emit('WASM_STATE_CHANGED', this._state);
    }
  }
}
