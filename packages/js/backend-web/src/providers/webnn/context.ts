/**
 * @fileoverview context.ts
 * Provides context functionality for the backend-web package.
 */
import { BackendError } from '@onnx9000/core';
/** Represents the `WebNNContextManager` class. */
export class WebNNContextManager {
  /** Property `instance` of `WebNNContextManager`. */
  private static instance: WebNNContextManager | null = null;
  /** Property `mlContext` of `WebNNContextManager`. */
  private mlContext: MLContext | null = null;
  /** Property `builder` of `WebNNContextManager`. */
  public builder: MLGraphBuilder | null = null;

  /** Initializes a new `WebNNContextManager` instance. */
  private constructor() {}

  /**
   * Executes `getInstance` on `WebNNContextManager`.
   * @returns Result of `getInstance`.
   */
  static getInstance(): WebNNContextManager {
    if (!WebNNContextManager.instance) {
      WebNNContextManager.instance = new WebNNContextManager();
    }
    return WebNNContextManager.instance;
  }

  /**
   * Executes `initialize` on `WebNNContextManager`.
   * @param options Parameter `options`.
   */
  async initialize(
    options: MLContextOptions = {
      deviceType: 'npu',
      powerPreference: 'default',
    },
  ): Promise<void> {
    if (typeof navigator === 'undefined' || !navigator.ml) {
      try {
        // Attempt to load the polyfill dynamically to avoid cyclic dependency
        const polyfillName = '@onnx9000/webnn-polyfill';
        await import(/* @vite-ignore */ polyfillName);
      } catch (e) {
        console.warn('Failed to load WebNN polyfill:', e);
      }
    }
    if (typeof navigator === 'undefined' || !navigator.ml) {
      throw new BackendError(
        'WebNN is not supported in this environment (navigator.ml is missing).',
      );
    }

    if (this.mlContext && this.builder) {
      return;
    }

    try {
      this.mlContext = await navigator.ml.createContext(options);
    } catch (e) {
      console.warn(
        `Failed to create WebNN context with options: ${JSON.stringify(options)}. Falling back to defaults.`,
        e,
      );
      try {
        this.mlContext = await navigator.ml.createContext();
      } catch (fallbackError) {
        throw new BackendError(`Failed to initialize WebNN context completely: ${fallbackError}`);
      }
    }

    if (typeof window !== 'undefined' && (window as ReturnType<typeof JSON.parse>).MLGraphBuilder) {
      this.builder = new MLGraphBuilder(this.mlContext);
    } else {
      const GlobalMLGraphBuilder = (globalThis as ReturnType<typeof JSON.parse>).MLGraphBuilder;
      if (GlobalMLGraphBuilder) {
        this.builder = new GlobalMLGraphBuilder(this.mlContext);
      } else {
        throw new BackendError('MLGraphBuilder is not available in this environment.');
      }
    }
  }

  /**
   * Executes `getContext` on `WebNNContextManager`.
   * @returns Result of `getContext`.
   */
  getContext(): MLContext {
    if (!this.mlContext) {
      throw new BackendError('WebNN Context is not initialized.');
    }
    return this.mlContext;
  }

  /**
   * Executes `getBuilder` on `WebNNContextManager`.
   * @returns Result of `getBuilder`.
   */
  getBuilder(): MLGraphBuilder {
    if (!this.builder) {
      throw new BackendError('WebNN MLGraphBuilder is not initialized.');
    }
    return this.builder;
  }

  /**
   * Executes `getCapabilities` on `WebNNContextManager`.
   * @returns Result of `getCapabilities`.
   */
  getCapabilities(): MLOpSupportLimits | null {
    if (!this.mlContext) return null;
    if (typeof this.mlContext.opSupportLimits === 'function') {
      return this.mlContext.opSupportLimits();
    }
    return null;
  }

  /** Executes `reset` on `WebNNContextManager`. */
  reset(): void {
    this.mlContext = null;
    this.builder = null;
  }
}
