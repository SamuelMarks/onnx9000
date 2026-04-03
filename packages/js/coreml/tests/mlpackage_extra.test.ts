import { describe, it, expect } from 'vitest';
import { MLPackageBuilder } from '../src/mlpackage.js';
import { Model } from '../src/schema.js';

describe('MLPackageBuilder', () => {
  it('covers all mlpackage options', async () => {
    const model: Model = {
      specificationVersion: 4,
      description: {
        input: [
          { name: 'in1', type: { tensorType: {} } as any },
          { name: 'in2' },
          { name: 'in3', type: { tensorType: {} } as any },
        ],
        output: [{ name: 'out1', type: { tensorType: {} } as any }, { name: 'out2' }],
      },
    };

    const options = {
      stateful: true,
      computePrecision: 'Float16' as any,
      imageInputs: { in1: { blueBias: 0.1 } },
      sequenceInputs: ['in2'],
      outputMappings: { out1: 'mapped_out' },
      classifierOutputs: ['out2'],
      classLabels: ['cat', 'dog'],
      vocabularyFiles: { 'vocab.txt': new Uint8Array([1, 2]) },
      generateSwiftBoilerplate: true,
      visionFrameworkDescription: 'MyVision',
    };

    const builder = new MLPackageBuilder(model, new Uint8Array([1, 2, 3]), options);

    class MockZip {
      files: Record<string, any> = {};
      file(name: string, data: any) {
        this.files[name] = data;
      }
      async generateAsync() {
        return new Uint8Array([0]);
      }
    }

    const bytes = await builder.createZipArchive(MockZip as any);
    expect(bytes).toBeInstanceOf(Uint8Array);
  });

  it('handles empty descriptions', async () => {
    const model: Model = { specificationVersion: 4 };
    const builder = new MLPackageBuilder(model, new Uint8Array(0));

    class MockZip {
      files: Record<string, any> = {};
      file(name: string, data: any) {
        this.files[name] = data;
      }
      async generateAsync() {
        return new Uint8Array([0]);
      }
    }

    const bytes = await builder.createZipArchive(MockZip as any);
    expect(bytes).toBeInstanceOf(Uint8Array);
  });

  it('handles falsy options for swift, stateful, image, unknown types', async () => {
    const model: Model = {
      specificationVersion: 4,
      description: {
        input: [{ name: 'in1' }],
        output: [{ name: 'out1' }],
      },
    };

    const options = {
      stateful: false,
      generateSwiftBoilerplate: true,
      visionFrameworkDescription: '',
    };
    const builder = new MLPackageBuilder(model, new Uint8Array(0), options);

    class MockZip {
      files: Record<string, any> = {};
      file(name: string, data: any) {
        this.files[name] = data;
      }
      async generateAsync() {
        return new Uint8Array([0]);
      }
    }
    const bytes = await builder.createZipArchive(MockZip as any);
    expect(bytes).toBeInstanceOf(Uint8Array);
  });
});
