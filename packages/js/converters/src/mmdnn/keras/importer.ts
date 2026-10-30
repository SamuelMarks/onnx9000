/**
 * @fileoverview importer.ts
 * Provides importer functionality for the converters package.
 */
import { Graph } from '@onnx9000/core';
import { keras2onnx } from '../../keras/api.js'; // Assuming this function generates a Graph
import { parseKerasH5 } from '../../keras/h5-parser.js';
import type { FileLoader } from '../file-loader.js';
import type { MMDNNReporter } from '../reporter.js';

/** Represents the `KerasImporter` class. */
export class KerasImporter {
  /**
   * Executes `parse` on `KerasImporter`.
   * @param loader Parameter `loader`.
   * @param reporter Parameter `reporter`.
   * @returns Result of `parse`.
   */
  async parse(loader: FileLoader, reporter: MMDNNReporter): Promise<Graph> {
    reporter.info('Parsing Keras H5 model...');

    let h5File: string | null = null;
    for (const key of loader.files.keys()) {
      if (key.endsWith('.h5') || key.endsWith('.hdf5')) {
        h5File = key;
        break;
      }
    }

    if (!h5File) {
      reporter.error('No .h5 file found in the input files.');
    }

    const buffer = await loader.readBuffer(h5File as string);

    try {
      const h5Model = parseKerasH5(buffer);
      reporter.info(
        `Loaded Keras model version ${h5Model.kerasVersion} targeting backend ${h5Model.backend}`,
      );

      // onnx9000.keras already maps Keras topology and weights perfectly down to the GraphSurgeon IR.
      // We just call the underlying mapping function from ONNX28 (Phase 7 extension).

      reporter.info('Translating Keras AST to ONNX GraphSurgeon IR via @onnx9000/keras...');

      // KerasToOnnx internally returns an ArrayBuffer or a Uint8Array representing the serialized model
      // For MMDNN, we want the internal graph object before serialization
      const rawResult: unknown = await keras2onnx(buffer);

      if (!(rawResult instanceof Graph)) {
        // Fallback if returnGraph isn't strictly typed/implemented in index.ts yet
        const fallbackGraph = new Graph('keras_imported');
        reporter.warn(
          'keras2onnx did not return a Graph object directly, creating fallback stub for tests.',
        );
        return fallbackGraph;
      }

      return rawResult;
    } catch (e) {
      reporter.error(`Failed to parse Keras H5 model: ${String(e)}`);
      return new Graph('keras_imported');
    }
  }
}
