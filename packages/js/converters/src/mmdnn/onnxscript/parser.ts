/* eslint-disable */
// @ts-nocheck
import { Graph, Node, ValueInfo } from '@onnx9000/core';

export class OnnxScriptParser {
  /**
   * Parses ONNXScript Python code into a basic ONNX IR Graph.
   * This is a lightweight text-based AST parser designed to operate
   * natively in the JS browser environment without the Pyodide/Python runtime overhead.
   */
  public parseScript(scriptContent: string): Graph {
    const graph = new Graph('onnxscript-imported');

    // Simplistic line-by-line regex parsing for our supported formats
    const lines = scriptContent.split('\n');
    let insideFunc = false;

    const ioRegex = /([a-zA-Z0-9_]+):\s*FLOAT(?:\[(.*?)\])?/g;

    for (let line of lines) {
      line = line.trim();

      // Look for function definition
      if (line.startsWith('def ')) {
        insideFunc = true;
        const sigMatch = line.match(/def\s+[a-zA-Z0-9_]+\s*\((.*?)\)(?:\s*->\s*(.*?))?:/);
        if (sigMatch) {
          const argsStr = sigMatch[1];
          const retStr = sigMatch[2];

          // Parse inputs
          let argMatch;
          if (argsStr) {
            while ((argMatch = ioRegex.exec(argsStr)) !== null) {
              const name = argMatch[1];
              const shape = argMatch[2]
                ? argMatch[2].split(',').map((s) => parseInt(s.trim(), 10))
                : [-1];
              if (name) {
                graph.inputs.push(new ValueInfo(name, shape, 'float32'));
              }
            }
          }

          // Parse output (if named, else assume 'Y')
          if (retStr && retStr.includes('FLOAT')) {
            const shapeStr = retStr.match(/FLOAT\[(.*?)\]/);
            const shape =
              shapeStr && shapeStr[1]
                ? shapeStr[1].split(',').map((s) => parseInt(s.trim(), 10))
                : [-1];
            // The parser doesn't natively know the output name until `return Y`
            // Will push output value info during return
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

        const node = new Node(opType, ins, outs, {}, `${opType}_${outs[0]}`);
        graph.nodes.push(node);
      }
    }

    return graph;
  }
}
