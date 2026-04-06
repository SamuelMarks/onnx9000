import { describe, it, expect, vi } from 'vitest';
import { extractTraceViaPyodide, PyodideInterface } from '../src/keras/autograph.js';

describe('Pyodide AutoGraph Tracing', () => {
  it('throws an error if no create_model() is provided', async () => {
    const mockPyodide: PyodideInterface = {
      runPythonAsync: async (code: string) => {
        // Mock a failure returning the JSON error format
        return JSON.stringify({
          error: "Provided Python code must define a 'create_model()' function.",
          traceback: 'Traceback (most recent call last):\n  ...',
        });
      },
      loadPackage: async () => {},
    };

    await expect(extractTraceViaPyodide(mockPyodide, 'pass', [10])).rejects.toThrow(
      "Provided Python code must define a 'create_model()' function.",
    );
  });

  it('extracts a mapped Functional model topology from traced python', async () => {
    const mockPyodide: PyodideInterface = {
      runPythonAsync: async (code: string) => {
        // Return a mock successful JSON response from the Python script
        return JSON.stringify({
          class_name: 'Functional',
          config: {
            name: 'TracedModel',
            layers: [
              {
                class_name: 'InputLayer',
                name: 'input_1',
                config: { name: 'input_1', batch_input_shape: [null, 10], dtype: 'float32' },
                inbound_nodes: [],
              },
              {
                class_name: 'MatMul',
                name: 'StatefulPartitionedCall/sequential/dense/MatMul',
                config: { name: 'StatefulPartitionedCall/sequential/dense/MatMul' },
                inbound_nodes: [[['input_1', 0, 0, {}]]],
              },
            ],
            input_layers: [['input_1', 0, 0]],
            output_layers: [['StatefulPartitionedCall/sequential/dense/MatMul', 0, 0]],
          },
        });
      },
      loadPackage: async () => {},
    };

    const code = `
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(32)

    def call(self, inputs):
        return self.dense(inputs)

def create_model():
    return MyModel()
`;

    const result = await extractTraceViaPyodide(mockPyodide, code, [10]);
    expect(result['class_name']).toBe('Functional');
    const config = result['config'] as Object;
    expect(config.name).toBe('TracedModel');
    expect(config.layers).toHaveLength(2);
    expect(config.layers[1].class_name).toBe('MatMul');
  });
});
