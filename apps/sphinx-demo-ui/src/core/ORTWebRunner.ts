import { WorkerManager } from './WorkerManager';
import { globalEventBus } from './EventBus';
import { Logger } from './Logger';

/**
 * Worker wrapper for ONNX Runtime Web execution.
 */
export class ORTWebRunner {
  private logger = Logger.getInstance();

  /**
   * Run inference on an ONNX model using ONNX Runtime Web.
   *
   * @param onnxBinary - The ONNX model binary.
   * @param inputs - The pre-processed Tensor inputs.
   * @param executionProvider - 'wasm' | 'webgl' | 'webgpu'
   * @returns A Promise resolving to a map of Output Tensors.
   */
  public async runInference(
    onnxBinary: Uint8Array,
    inputs: Record<string, Float32Array | Int32Array>,
    executionProvider: 'wasm' | 'webgl' | 'webgpu' = 'wasm'
  ): Promise<Record<string, any>> {
    console.log(`Starting ORT Web execution (${executionProvider})...`);
    globalEventBus.emit('INFERENCE_STARTED');

    const start = performance.now();

    try {
      WorkerManager.getInstance().initWorker('/workers/ort-worker.js');
      const outputs = (await WorkerManager.getInstance().execute('RUN_ORT', {
        binary: onnxBinary,
        inputs,
        executionProvider
      })) as Record<string, any>;

      const end = performance.now();
      const latency = end - start;
      console.log(`ORT Web execution complete in ${latency.toFixed(2)} ms.`);

      globalEventBus.emit('INFERENCE_SUCCESS', { latency, outputs });
      return outputs;
    } catch (e: any) {
      console.error(`ORT Web execution failed`, e);
      globalEventBus.emit('INFERENCE_ERROR', e);
      throw e;
    } finally {
      WorkerManager.getInstance().terminate();
    }
  }
}
