/**
 * @fileoverview CoreML Converter Demo UI.
 */
import { Graph, Node, ValueInfo } from '@onnx9000/core';
import { convertToCoreML } from '@onnx9000/coreml';

/**
 * Formats a value for CoreML MIL AST JSON serialization, converting `bigint` to string.
 * @param _key Property key.
 * @param value Property value.
 * @returns Serialized value.
 */
export function coremlJsonReplacer(_key: string, value: unknown): unknown {
  if (typeof value === 'bigint') {
    return `${value.toString()}n`;
  }
  return value;
}

/**
 * Initializes the CoreML converter demo UI.
 */
export function initCoremlDemo(): void {
  const convertBtn = document.getElementById('convert-btn') as HTMLButtonElement | null;
  const out = document.getElementById('output') as HTMLElement | null;

  if (!convertBtn || !out) return;

  convertBtn.addEventListener('click', () => {
    out.innerText = 'Converting...';

    const g = new Graph('mock_model');
    g.inputs.push(new ValueInfo('input1', [1, 3, 224, 224], 'float32'));
    g.outputs.push(new ValueInfo('output1', [1, 1000], 'float32'));

    const reluNode = new Node('Relu', ['input1'], ['output1']);
    g.nodes.push(reluNode);

    const milAst = convertToCoreML(g);
    const jsonString = JSON.stringify(milAst, coremlJsonReplacer, 2);
    out.innerText = jsonString;
  });
}

initCoremlDemo();
