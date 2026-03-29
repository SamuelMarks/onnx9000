/* eslint-disable */
// @ts-nocheck
import { WorkerManager } from './WorkerManager';
import { globalEventBus } from './EventBus';
import { Store } from './Store';
import { Logger } from './Logger';

/**
 * Worker wrapper for Olive and onnx-simplifier execution.
 * Interfaces with the WASM builds to optimize ONNX graphs.
 */
export class OliveOptimizer {
  private logger = Logger.getInstance();

  /**
   * Optimizes the ONNX binary graph using the specified configuration.
   *
   * @param onnxBinary - The raw ONNX binary representation.
   * @param config - The Olive configuration options.
   * @returns A Promise resolving to the optimized ONNX binary.
   */
  public async optimize(
    onnxBinary: Uint8Array,
    config: {
      quantizationLevel: 'FP16' | 'INT8' | 'None';
      enableStaticShapeInference: boolean;
      enableTransformerFusion: boolean;
    }
  ): Promise<Uint8Array> {
    console.log(`Starting Olive optimization...`, config);
    globalEventBus.emit('OLIVE_OPTIMIZATION_STARTED');

    const originalSize = onnxBinary.byteLength;
    console.log(`Original size: ${(originalSize / 1024).toFixed(2)} KB`);

    try {
      WorkerManager.getInstance().initWorker('/workers/olive-worker.js');
      const optimizedBinary = (await WorkerManager.getInstance().execute('OPTIMIZE_ONNX', {
        binary: onnxBinary,
        config: config
      })) as Uint8Array;

      const newSize = optimizedBinary.byteLength;
      console.log(`Optimized size: ${(newSize / 1024).toFixed(2)} KB`);
      const ratio = (((originalSize - newSize) / originalSize) * 100).toFixed(2);
      console.log(`Compression ratio: ${ratio}% reduction`);

      globalEventBus.emit('OLIVE_COMPRESSION_METRIC', { originalSize, newSize, ratio });
      globalEventBus.emit('OLIVE_OPTIMIZATION_SUCCESS');

      return optimizedBinary;
    } catch (e: object) {
      console.error(`Olive optimization failed`, e);
      globalEventBus.emit('OLIVE_OPTIMIZATION_ERROR', e);
      throw e;
    } finally {
      WorkerManager.getInstance().terminate();
    }
  }

  /**
   * Run purely onnx-simplifier on the graph.
   */
  public async simplify(onnxBinary: Uint8Array): Promise<Uint8Array> {
    console.log(`Starting ONNX simplification...`);
    globalEventBus.emit('ONNX_SIMPLIFIER_STARTED');

    try {
      WorkerManager.getInstance().initWorker('/workers/simplifier-worker.js');
      const simplifiedBinary = (await WorkerManager.getInstance().execute('SIMPLIFY_ONNX', {
        binary: onnxBinary
      })) as Uint8Array;
      globalEventBus.emit('ONNX_SIMPLIFIER_SUCCESS');
      return simplifiedBinary;
    } catch (e: object) {
      console.error(`ONNX simplification failed`, e);
      globalEventBus.emit('ONNX_SIMPLIFIER_ERROR', e);
      throw e;
    } finally {
      WorkerManager.getInstance().terminate();
    }
  }
}
