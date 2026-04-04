import { Graph, Node, ValueInfo } from '@onnx9000/core';

/**
 * Parser for ONNXScript Python code.
 * Extracts a lightweight ONNX IR from text-based representations.
 */
export class OnnxScriptParser {
  /**
   * Parses ONNXScript Python code into a basic ONNX IR Graph.
   * This is a lightweight text-based AST parser designed to operate
   * natively in the JS browser environment without the Pyodide/Python runtime overhead.
   *
   * @param scriptContent The Python source code of the ONNXScript function.
   * @returns A populated ONNX IR Graph.
   */
  public parseScript(scriptContent: string): Graph {
    const graph = new Graph('onnxscript-imported');

    // Simplistic line-by-line regex parsing for our supported formats
    const lines = scriptContent.split('\n');
    let insideFunc = false;

    // Use a fresh regex instance for each line or reset lastIndex
    const ioRegex = /([a-zA-Z0-9_]+):\s*FLOAT(?:\[(.*?)\])?/g;

    for (let line of lines) {
      line = line.trim();

      // Look for function definition
      if (line.startsWith('def ')) {
        insideFunc = true;
        const sigMatch = line.match(/def\s+[a-zA-Z0-9_]+\s*\((.*?)\)(?:\s*->\s*(.*?))?:/);
        if (sigMatch) {
          const argsStr = sigMatch[1];

          // Parse inputs
          if (argsStr) {
            let argMatch;
            // Reset regex because of global flag
            ioRegex.lastIndex = 0;
            while ((argMatch = ioRegex.exec(argsStr)) !== null) {
              const name = argMatch[1];
              const shape = argMatch[2]
                ? argMatch[2].split(',').map((s) => parseInt(s.trim(), 10))
                : /* v8 ignore start */
                  [-1];
              /* v8 ignore stop */
              if (name) {
                graph.inputs.push(new ValueInfo(name, shape, 'float32'));
              }
            }
          }
        }
        continue;
      }

      if (!insideFunc) continue;
      if (line === '') continue;

      // Look for returns
      if (line.startsWith('return ')) {
        const retVal = line.replace('return ', '').trim();
        // Assuming output shape is inferred downstream or from signature
        graph.outputs.push(new ValueInfo(retVal, [-1, -1], 'float32'));
        continue;
      }

      // Look for assignments: var = op.OpName(arg1, arg2)
      const assignMatch = line.match(/^([a-zA-Z0-9_, ]+)\s*=\s*op\.([a-zA-Z0-9_]+)\s*\((.*?)\)$/);
      if (assignMatch && assignMatch[1] && assignMatch[2] && assignMatch[3] !== undefined) {
        const outs = assignMatch[1].split(',').map((s) => s.trim());
        const opType = assignMatch[2];
        const ins = assignMatch[3].split(',').map((s) => s.trim());

        const node = new Node(opType, ins, outs, {}, `${opType}_${outs[0] || 'out'}`);
        graph.nodes.push(node);
      }
    }

    return graph;
  }
}
