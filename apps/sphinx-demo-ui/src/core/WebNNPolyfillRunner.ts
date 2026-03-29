import { WorkerManager } from './WorkerManager';
import { globalEventBus } from './EventBus';
import { Logger } from './Logger';

/**
 * Worker wrapper for WebNN polyfill execution.
 */
export class WebNNPolyfillRunner {
  private logger = Logger.getInstance();

  /**
   * Run inference on an ONNX model using WebNN polyfill.
   *
   * @param onnxBinary - The ONNX model binary.
   * @param inputs - The pre-processed Tensor inputs.
   * @returns A Promise resolving to a map of Output Tensors.
   */
  public async runInference(
    onnxBinary: Uint8Array,
    inputs: Record<string, Float32Array | Int32Array>
  ): Promise<Record<string, any>> {
    console.log(`Starting WebNN Polyfill execution...`);
    globalEventBus.emit('INFERENCE_STARTED');

    const start = performance.now();

    try {
      WorkerManager.getInstance().initWorker('/workers/webnn-worker.js');
      const outputs = (await WorkerManager.getInstance().execute('RUN_WEBNN', {
        binary: onnxBinary,
        inputs
      })) as Record<string, any>;

      const end = performance.now();
      const latency = end - start;
      console.log(`WebNN execution complete in ${latency.toFixed(2)} ms.`);

      globalEventBus.emit('INFERENCE_SUCCESS', { latency, outputs });
      return outputs;
    } catch (e: any) {
      console.error(`WebNN execution failed`, e);
      globalEventBus.emit('INFERENCE_ERROR', e);
      throw e;
    } finally {
      WorkerManager.getInstance().terminate();
    }
  }
}
