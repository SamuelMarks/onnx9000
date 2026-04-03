import { describe, it, expect, vi } from 'vitest';
import { MLPackageLoader } from '../src/loader.js';
import { emitModel } from '../src/emitter.js';

describe('MLPackageLoader', () => {
  it('loads from zip correctly', async () => {
    const model = { specificationVersion: 1 };
    const modelBytes = emitModel(model as any);

    class MockZip {
      files: Record<string, any> = {};
      async loadAsync() {
        this.files['Data/com.apple.CoreML/model.mlmodel'] = {
          async: async () => modelBytes,
        };
        this.files['Data/com.apple.CoreML/weights/weight.bin'] = {
          async: async () => new Uint8Array([1, 2, 3]),
        };
      }
      file(name: string) {
        return this.files[name];
      }
    }

    const { model: loaded, weights } = await MLPackageLoader.loadFromZip(
      MockZip,
      new Uint8Array(0),
    );
    expect(loaded.specificationVersion).toBe(1);
    expect(weights.length).toBe(3);

    // No weights
    class MockZipNoWeights extends MockZip {
      async loadAsync() {
        this.files['Data/com.apple.CoreML/model.mlmodel'] = {
          async: async () => modelBytes,
        };
      }
    }
    const { weights: emptyWeights } = await MLPackageLoader.loadFromZip(
      MockZipNoWeights,
      new Uint8Array(0),
    );
    expect(emptyWeights.length).toBe(0);

    // No model
    class MockZipNoModel extends MockZip {
      async loadAsync() {}
    }
    await expect(MLPackageLoader.loadFromZip(MockZipNoModel, new Uint8Array(0))).rejects.toThrow(
      'model.mlmodel not found in package',
    );
  });

  it('loads ast stub', () => {
    const prog = MLPackageLoader.parseMILProgram({
      mlProgram: { version: 1, functions: {} },
      specificationVersion: 1,
    });
    expect(prog).toBeDefined();
    expect(prog.functions['main']).toBeDefined();

    const prog2 = MLPackageLoader.parseMILProgram({
      neuralNetwork: { layers: [] },
      specificationVersion: 1,
    });
    expect(prog2).toBeDefined();
  });
});
