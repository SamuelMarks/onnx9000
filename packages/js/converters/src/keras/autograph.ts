import { JsonObject, JsonArray } from './tfjs-parser.js';

export interface PyodideInterface {
  runPythonAsync: (code: string) => Promise<string>;
  loadPackage: (names: string | string[]) => Promise<void>;
}

/**
 * Resolves Keras `Model` subclassing topologies by falling back to execution-trace extraction
 * via Pyodide/Python `tf.autograph` tracing.
 *
 * This injects the provided Python model definition into a WebAssembly Python environment,
 * initializes the model, performs a symbolic trace using `tf.function`, and extracts the
 * underlying GraphDef. The GraphDef is then normalized back into a standard Keras Functional AST.
 *
 * @param pyodide The initialized Pyodide WASM instance.
 * @param modelCode The raw Python source code defining the model. Must include a `create_model()` function that returns the instantiated subclassed model.
 * @param inputShape The expected input shape (excluding batch dimension).
 * @returns A JSON object mimicking a standard Keras `Functional` topology that can be parsed by `extractKerasTopology`.
 */
export async function extractTraceViaPyodide(
  pyodide: PyodideInterface,
  modelCode: string,
  inputShape: (number | null)[],
): Promise<JsonObject> {
  // We stringify the shape, converting JS nulls to Python Nones
  const shapeStr = JSON.stringify(inputShape).replace(/null/g, 'None');

  const pythonScript = `
import json
import traceback

def trace_model():
    try:
        import tensorflow as tf
        
        # User-provided model definition injected into local scope
        user_globals = {}
        exec("""
${modelCode.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}
        """, user_globals)
        
        if 'create_model' not in user_globals:
            raise ValueError("Provided Python code must define a 'create_model()' function.")
            
        model = user_globals['create_model']()
        
        # Ensure the model is built by calling it with dummy data or tracing it
        @tf.function(input_signature=[tf.TensorSpec(shape=[None] + ${shapeStr}, dtype=tf.float32)])
        def traced_forward(x):
            return model(x)
            
        concrete_func = traced_forward.get_concrete_function()
        graph_def = concrete_func.graph.as_graph_def()
        
        # Map TF GraphDef to a pseudo Keras Functional Config
        # This translates raw TF operations back into an AST that KerasModelTopology can ingest.
        layers = []
        
        # Insert an explicit InputLayer
        layers.append({
            "class_name": "InputLayer",
            "name": "input_1",
            "config": {
                "name": "input_1",
                "batch_input_shape": [None] + ${shapeStr},
                "dtype": "float32"
            },
            "inbound_nodes": []
        })
        
        output_layer_names = []
        
        for node in graph_def.node:
            # Skip internal TF constants/variables for the simplified AST trace
            if node.op in ['Const', 'ReadVariableOp', 'ResourceGather', 'VarHandleOp']:
                continue
                
            inbound_nodes = []
            for inp in node.input:
                # Strip tensor indices for simple dependency tracking
                clean_inp = inp.split(':')[0]
                if clean_inp == 'x': # Maps to our defined input
                    inbound_nodes.append(['input_1', 0, 0, {}])
                elif clean_inp:
                    inbound_nodes.append([clean_inp, 0, 0, {}])
            
            layers.append({
                "class_name": node.op, # The mapped operation (e.g., MatMul, Add, Conv2D)
                "name": node.name,
                "config": {"name": node.name},
                "inbound_nodes": [inbound_nodes] if inbound_nodes else []
            })
            output_layer_names.append(node.name)
            
        # The last node added is generally the output of the model
        final_output = output_layer_names[-1] if output_layer_names else "input_1"

        return json.dumps({
            "class_name": "Functional",
            "config": {
                "name": "TracedModel",
                "layers": layers,
                "input_layers": [["input_1", 0, 0]],
                "output_layers": [[final_output, 0, 0]]
            }
        })
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})

trace_model()
`;

  const jsonStr = await pyodide.runPythonAsync(pythonScript);
  const parsed = JSON.parse(jsonStr) as JsonObject;
  
  if (parsed['error']) {
    throw new Error(`Pyodide Trace Error: ${parsed['error'] as string}\n${parsed['traceback'] as string}`);
  }

  return parsed;
}
