import { describe, it, expect } from 'vitest';
import { parseTFJSModel } from '../src/keras/tfjs-parser.js';

describe('tfjs-parser', () => {
  it('should parse layers-model correctly with all fields', () => {
    const json = {
      format: 'layers-model',
      generatedBy: 'Keras v2.3.0',
      convertedBy: 'TensorFlow.js Converter v1.0.0',
      modelTopology: {
        class_name: 'Sequential',
        config: { name: 'test_model', layers: [] },
      },
      weightsManifest: [
        {
          paths: ['weights.bin'],
          weights: [
            {
              name: 'dense/kernel',
              shape: [2, 2],
              dtype: 'float32',
              quantization: { scale: 0.1, min: 0.0, dtype: 'uint8' },
            },
          ],
        },
      ],
    };
    const model = parseTFJSModel(JSON.stringify(json));
    expect(model.format).toBe('layers-model');
    expect(model.generatedBy).toBe('Keras v2.3.0');
    expect(model.convertedBy).toBe('TensorFlow.js Converter v1.0.0');
    expect(model.weightsManifest[0].paths[0]).toBe('weights.bin');
    expect(model.weightsManifest[0].weights[0].name).toBe('dense/kernel');
    expect(model.weightsManifest[0].weights[0].quantization?.scale).toBe(0.1);
  });

  it('should fall back to graph-model based on topology', () => {
    const json = {
      modelTopology: {
        node: [{ name: 'Const', op: 'Const' }],
      },
      weightsManifest: [],
    };
    const model = parseTFJSModel(JSON.stringify(json));
    expect(model.format).toBe('graph-model');
  });

  it('should fall back to layers-model based on topology class_name', () => {
    const json = {
      modelTopology: {
        class_name: 'Model',
      },
      weightsManifest: [],
    };
    const model = parseTFJSModel(JSON.stringify(json));
    expect(model.format).toBe('layers-model');
  });

  it('should throw for unrecognized format', () => {
    const json = {
      modelTopology: {
        unknown: true,
      },
      weightsManifest: [],
    };
    expect(() => parseTFJSModel(JSON.stringify(json))).toThrow(
      'Unsupported or unrecognized TF.js model format',
    );
  });
});
