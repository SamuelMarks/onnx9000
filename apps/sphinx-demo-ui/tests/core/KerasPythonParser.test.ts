/* eslint-disable */
// @ts-nocheck
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { KerasPythonParser } from '../../src/core/KerasPythonParser';

let pyodideInstance: object = null;
(window as object).loadPyodide = vi.fn().mockImplementation(async () => {
  if (pyodideInstance) return pyodideInstance;

  const state: object = { _keras_parsed_model: null };

  pyodideInstance = {
    runPythonAsync: vi.fn().mockImplementation(async (code: string) => {
      if (code.includes('mock_keras = ModuleType')) {
        // Setup mock
        return;
      }
      if (code.includes('_keras_parsed_model = None')) {
        state._keras_parsed_model = null;
        return;
      }
      if (code.includes('json.dumps(_keras_parsed_model)')) {
        if (state._keras_parsed_model === 'None' || !state._keras_parsed_model) {
          throw new Error('Could not find models.Sequential in the Python code.');
        }
        return JSON.stringify(state._keras_parsed_model);
      }

      // User code mock execution logic
      if (code.includes('models.Sequential')) {
        state._keras_parsed_model = {
          format: 'layers-model',
          modelTopology: {
            class_name: 'Sequential',
            config: { layers: [] }
          }
        };

        // Inject layers based on mocked user code
        if (code.includes('Conv2D(32')) {
          state._keras_parsed_model.modelTopology.config.layers.push({
            class_name: 'InputLayer',
            config: { batch_input_shape: [null, 28, 28, 1] }
          });
          state._keras_parsed_model.modelTopology.config.layers.push({
            class_name: 'Conv2D',
            config: { filters: 32 }
          });
          state._keras_parsed_model.modelTopology.config.layers.push({
            class_name: 'MaxPooling2D',
            config: {}
          });
          state._keras_parsed_model.modelTopology.config.layers.push({
            class_name: 'Flatten',
            config: {}
          });
          state._keras_parsed_model.modelTopology.config.layers.push({
            class_name: 'Dense',
            config: { units: 128 }
          });
          state._keras_parsed_model.modelTopology.config.layers.push({
            class_name: 'Dense',
            config: { units: 10 }
          });
        } else if (code.includes('Conv2D()')) {
          // Empty args
        } else if (code.includes('Conv2D(64)')) {
          state._keras_parsed_model.modelTopology.config.layers.push({
            class_name: 'Conv2D',
            config: { filters: 64 }
          });
        }
      }
      return;
    })
  };
  return pyodideInstance;
});

describe('KerasPythonParser', () => {
  beforeEach(() => {
    (KerasPythonParser as object).pyodideInstance = null;
    (KerasPythonParser as object).isLoading = false;
  });

  it('should parse a simple Keras Sequential model', async () => {
    const code = `
model = models.Sequential([
    layers.Input(shape=(28, 28, 1), name='image_input'),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax', name='output_probs')
])
    `;
    const parsed = await KerasPythonParser.parse(code);
    expect(parsed.format).toBe('layers-model');
    expect(parsed.modelTopology.config.layers.length).toBe(6);
    expect(parsed.modelTopology.config.layers[0].class_name).toBe('InputLayer');
  });

  it('should throw if no sequential model found', async () => {
    await expect(KerasPythonParser.parse('print("hello")')).rejects.toThrow(
      'Could not find models.Sequential'
    );
  });

  it('should parse layers without arguments', async () => {
    const code = `
model = models.Sequential([
    layers.Input(),
    layers.Conv2D(),
    layers.MaxPooling2D(),
    layers.Dense()
])
    `;
    const parsed = await KerasPythonParser.parse(code);
    expect(parsed.format).toBe('layers-model');
  });

  it('should parse Conv2D without kernel tuple', async () => {
    const code = `
model = models.Sequential([
    layers.Conv2D(64),
    layers.Dense(name='no_units')
])
    `;
    const parsed = await KerasPythonParser.parse(code);
    expect(parsed.format).toBe('layers-model');
  });

  it('should handle pending loads correctly', async () => {
    const p1 = KerasPythonParser.initPyodide();
    const p2 = KerasPythonParser.initPyodide();
    await Promise.all([p1, p2]);
  });
});

it('should handle pending loads correctly with while loop delay', async () => {
  (KerasPythonParser as object).isLoading = true;
  setTimeout(() => {
    (KerasPythonParser as object).pyodideInstance = {};
  }, 150);
  const p1 = await KerasPythonParser.initPyodide();
  expect(p1).toBeTruthy();
});

it('should catch load error', async () => {
  // Override mock to throw
  (window as object).loadPyodide.mockRejectedValueOnce(new Error('Network error'));

  (KerasPythonParser as object).pyodideInstance = null;
  (KerasPythonParser as object).isLoading = false;

  await expect(KerasPythonParser.initPyodide()).rejects.toThrow('Network error');
});
