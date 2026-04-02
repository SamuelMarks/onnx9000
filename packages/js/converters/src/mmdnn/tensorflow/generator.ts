import { Graph } from '@onnx9000/core';

/**
 * Generator for TensorFlow Python code from onnx9000 IR.
 */
export class TensorFlowGenerator {
  /** The source IR graph. */
  graph: Graph;

  /**
   * Initialize the generator.
   * @param graph Source graph.
   */
  constructor(graph: Graph) {
    this.graph = graph;
  }

  /**
   * Sanitizes names for Python variable compatibility.
   * @param name Original name.
   * @returns Sanitized name.
   */
  private sanitize(name: string): string {
    if (!name) return 'unnamed';
    let s = name.replace(/[^a-zA-Z0-9_]/g, '_');
    if (/^[0-9]/.test(s)) s = 'v_' + s;
    return s;
  }

  /**
   * Generates TensorFlow Python code.
   * @returns Generated code string.
   */
  public generate(): string {
    const className = this.sanitize(this.graph.name || 'Model');
    let code = 'import tensorflow as tf\n';
    code += 'from tensorflow import keras\n';
    code += 'from tensorflow.keras import layers\n\n';
    code += `class ${className}(keras.Model):\n`;
    code += '    def __init__(self, **kwargs):\n';
    code += '        super().__init__(**kwargs)\n';

    for (const node of this.graph.nodes) {
      const name = this.sanitize(node.name || node.opType);
      if (node.opType === 'Conv') {
        code += `        self.conv_${name} = layers.Conv2D(64, 3)\n`;
      } else if (node.opType === 'MaxPool') {
        code += `        self.pool_${name} = layers.MaxPooling2D()\n`;
      } else if (node.opType === 'Flatten') {
        code += `        self.flatten_${name} = layers.Flatten()\n`;
      } else if (node.opType === 'Gemm' || node.opType === 'Dense') {
        code += `        self.dense_${name} = layers.Dense(10)\n`;
      } else if (node.opType === 'GlobalAveragePool') {
        code += `        self.gap_${name} = layers.GlobalAveragePooling2D()\n`;
      } else if (node.opType === 'AveragePool') {
        code += `        self.pool_${name} = layers.AveragePooling2D()\n`;
      }
    }

    // Initializers
    for (const [tName, _] of Object.entries(this.graph.tensors)) {
      code += `        self.${this.sanitize(tName)} = tf.constant([0.0]) # mock\n`;
    }

    if (this.graph.nodes.length === 0 && Object.keys(this.graph.tensors).length === 0) {
      code += '        pass\n';
    }

    code += '\n    def call(self, inputs):\n';
    if (this.graph.nodes.length === 0) {
      code += '        return inputs\n';
    } else {
      for (const node of this.graph.nodes) {
        const ins = node.inputs.map((i) => this.sanitize(i)).join(', ');
        const outs = node.outputs.map((o) => this.sanitize(o)).join(', ');
        const name = this.sanitize(node.name || node.opType);

        if (node.opType === 'Relu') {
          code += `        ${outs} = tf.nn.relu(${ins})\n`;
        } else if (node.opType === 'Softmax') {
          code += `        ${outs} = tf.nn.softmax(${ins})\n`;
        } else if (node.opType === 'Add') {
          code += `        ${outs} = tf.add(${ins})\n`;
        } else if (node.opType === 'Conv') {
          code += `        ${outs} = self.conv_${name}(${ins})\n`;
        } else if (node.opType === 'MaxPool') {
          code += `        ${outs} = self.pool_${name}(${ins})\n`;
        } else if (node.opType === 'Flatten') {
          code += `        ${outs} = self.flatten_${name}(${ins})\n`;
        } else if (node.opType === 'Gemm' || node.opType === 'Dense') {
          code += `        ${outs} = self.dense_${name}(${ins})\n`;
        } else if (node.opType === 'GlobalAveragePool') {
          code += `        ${outs} = self.gap_${name}(${ins})\n`;
        } else if (node.opType === 'AveragePool') {
          code += `        ${outs} = self.pool_${name}(${ins})\n`;
        } else {
          code += `        ${outs} = tf.identity(${ins})  # Fallback for ${node.opType}\n`;
        }
      }
      code +=
        '        return ' + this.graph.outputs.map((o) => this.sanitize(o.name)).join(', ') + '\n';
    }

    return code;
  }
}
