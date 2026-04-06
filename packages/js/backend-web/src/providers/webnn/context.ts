export class WebNNContextManager {
  private static instance: WebNNContextManager | null = null;
  private mlContext: MLContext | null = null;
  public builder: MLGraphBuilder | null = null;

  private constructor() {}

  static getInstance(): WebNNContextManager {
    if (!WebNNContextManager.instance) {
      WebNNContextManager.instance = new WebNNContextManager();
    }
    return WebNNContextManager.instance;
  }

  async initialize(
    options: MLContextOptions = { deviceType: 'npu', powerPreference: 'default' },
  ): Promise<void> {
    if (typeof navigator === 'undefined' || !navigator.ml) {
      try {
        // Attempt to load the polyfill dynamically to avoid cyclic dependency
        const polyfillName = '@onnx9000/webnn-polyfill';
        // @ts-ignore: Dynamic import of polyfill
        await import(/* @vite-ignore */ polyfillName);
      } catch (e) {
        console.warn('Failed to load WebNN polyfill:', e);
      }
    }
    if (typeof navigator === 'undefined' || !navigator.ml) {
      throw new Error('WebNN is not supported in this environment (navigator.ml is missing).');
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
        throw new Error(`Failed to initialize WebNN context completely: ${fallbackError}`);
      }
    }

    if (typeof window !== 'undefined' && (window as ReturnType<typeof JSON.parse>).MLGraphBuilder) {
      this.builder = new MLGraphBuilder(this.mlContext);
    } else {
      const GlobalMLGraphBuilder = (globalThis as ReturnType<typeof JSON.parse>).MLGraphBuilder;
      if (GlobalMLGraphBuilder) {
        this.builder = new GlobalMLGraphBuilder(this.mlContext);
      } else {
        throw new Error('MLGraphBuilder is not available in this environment.');
      }
    }
  }

  getContext(): MLContext {
    if (!this.mlContext) {
      throw new Error('WebNN Context is not initialized.');
    }
    return this.mlContext;
  }

  getBuilder(): MLGraphBuilder {
    if (!this.builder) {
      throw new Error('WebNN MLGraphBuilder is not initialized.');
    }
    return this.builder;
  }

  getCapabilities(): MLOpSupportLimits | null {
    if (!this.mlContext) return null;
    if (typeof this.mlContext.opSupportLimits === 'function') {
      return this.mlContext.opSupportLimits();
    }
    return null;
  }

  reset(): void {
    this.mlContext = null;
    this.builder = null;
  }
}
