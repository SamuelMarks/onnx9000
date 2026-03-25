type PyodideInterface = any;

/**
 * Utility to parse simple Keras Python scripts into Keras model.json format
 * for the interactive demo. It lazily loads Pyodide on the first execution
 * to adhere to the lightweight initial load constraint, avoiding massive
 * downloads unless Python is actually utilized.
 */
export class KerasPythonParser {
  private static pyodideInstance: PyodideInterface | null = null;
  private static isLoading = false;

  /**
   * Initializes Pyodide if it hasn't been already.
   */
  public static async initPyodide(): Promise<PyodideInterface> {
    if (this.pyodideInstance) {
      return this.pyodideInstance;
    }

    if (this.isLoading) {
      // Wait for the instance to be loaded
      while (!this.pyodideInstance) {
        await new Promise((resolve) => setTimeout(resolve, 100));
      }
      return this.pyodideInstance;
    }

    this.isLoading = true;
    console.log('[stdout] Fetching Pyodide runtime (lazily loaded on first Python execution)...');

    try {
      if (!(window as any).loadPyodide) {
        // Dynamically inject script
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js';
        document.head.appendChild(script);
        await new Promise((resolve, reject) => {
          script.onload = resolve;
          script.onerror = reject;
        });
      }
      this.pyodideInstance = await (window as any).loadPyodide({
        indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/'
      });
      console.log('[stdout] Pyodide runtime loaded successfully.');
    } catch (e: any) {
      console.error('[stderr] Pyodide initialization failed:', e);
      throw e;
    } finally {
      this.isLoading = false;
    }

    return this.pyodideInstance;
  }

  /**
   * Parses the Python code and extracts the Keras Sequential model definition
   * into an equivalent JSON representation.
   *
   * @param pythonCode The source code string containing the Python Keras model.
   * @returns An object representing the Keras layers model topology.
   * @throws {Error} If no models.Sequential definition is found.
   */
  public static async parse(pythonCode: string): Promise<any> {
    const pyodide = await this.initPyodide();

    // We inject a mocked 'keras' and 'tensorflow' environment into Pyodide.
    // This allows the python code to "run" and describe its shape without downloading
    // the massive 200MB+ tensorflow package. The mock intercepts the calls and
    // builds an exact JSON representation mapping to the keras `layers-model`.
    const mockScript = `
import sys
import json
from types import ModuleType

class MockLayer:
    def __init__(self, class_name, **kwargs):
        self.class_name = class_name
        self.config = kwargs.copy()
        
        # Add basic name
        if 'name' not in self.config:
            self.config['name'] = f"{class_name.lower()}_{id(self)}"

    def to_dict(self):
        return {
            "class_name": self.class_name,
            "config": self.config
        }

class MockModels:
    @staticmethod
    def Sequential(layers=None):
        layers = layers or []
        config = {
            "name": "sequential_model",
            "layers": [l.to_dict() for l in layers]
        }
        
        # Cache the parsed data globally for extraction
        global _keras_parsed_model
        _keras_parsed_model = {
            "format": "layers-model",
            "modelTopology": {
                 "class_name": "Sequential",
                 "config": config
            },
            "weightsManifest": []
        }
        
        class _ModelInstance:
            def compile(self, *args, **kwargs): pass
            def fit(self, *args, **kwargs): pass
            def save(self, *args, **kwargs): pass
        
        return _ModelInstance()

class MockLayers:
    @staticmethod
    def Input(**kwargs):
        if 'shape' in kwargs:
            kwargs['batch_input_shape'] = [None] + list(kwargs.pop('shape'))
        return MockLayer("InputLayer", **kwargs)
    @staticmethod
    def Conv2D(*args, **kwargs):
        if len(args) > 0: kwargs['filters'] = args[0]
        if len(args) > 1: kwargs['kernel_size'] = args[1]
        return MockLayer("Conv2D", **kwargs)
    @staticmethod
    def MaxPooling2D(*args, **kwargs):
        if len(args) > 0: kwargs['pool_size'] = args[0]
        return MockLayer("MaxPooling2D", **kwargs)
    @staticmethod
    def Flatten(**kwargs): return MockLayer("Flatten", **kwargs)
    @staticmethod
    def Dense(*args, **kwargs):
        if len(args) > 0: kwargs['units'] = args[0]
        return MockLayer("Dense", **kwargs)

# Construct module structure
mock_keras = ModuleType('keras')
mock_keras.models = MockModels
mock_keras.layers = MockLayers
sys.modules['keras'] = mock_keras
sys.modules['keras.models'] = mock_keras.models
sys.modules['keras.layers'] = mock_keras.layers

mock_tf = ModuleType('tensorflow')
mock_tf.keras = mock_keras
sys.modules['tensorflow'] = mock_tf
sys.modules['tensorflow.keras'] = mock_keras
sys.modules['tensorflow.keras.models'] = mock_keras.models
sys.modules['tensorflow.keras.layers'] = mock_keras.layers
`;

    // 1. Initialize mock environment
    await pyodide.runPythonAsync(mockScript);

    // 2. Clear previous parse
    await pyodide.runPythonAsync('_keras_parsed_model = None');

    // 3. Run user code
    await pyodide.runPythonAsync(pythonCode);

    // 4. Extract JSON
    const extractScript = `
if _keras_parsed_model is None:
    raise RuntimeError("Could not find models.Sequential in the Python code.")
json.dumps(_keras_parsed_model)
`;
    const resultJson = await pyodide.runPythonAsync(extractScript);
    return JSON.parse(resultJson);
  }
}
