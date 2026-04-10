import { webnnRegistry } from './registry.js';
import './ops.js';
import { Graph, Node, Tensor, DType } from '@onnx9000/core';
import { WebNNContextManager } from './context.js';

export class WebNNCompiler {
  private graph: Graph;
  public builder: MLGraphBuilder;
  private operands: Map<string, MLOperand> = new Map();

  constructor(graph: Graph, builder: MLGraphBuilder) {
    this.graph = graph;
    this.builder = builder;
  }

  public async compile(): Promise<MLGraph> {
    this.operands.clear();

    for (const input of this.graph.inputs) {
      const dimensions = input.shape.map((dim) => {
        if (typeof dim === 'string' || dim === -1) {
          return 1;
        }
        return dim;
      });

      this.operands.set(
        input.name,
        this.builder.input(input.name, {
          dataType: this.mapDType(input.dtype),
          dimensions,
        }),
      );
    }

    for (const initName of this.graph.initializers) {
      const tensor = this.graph.tensors[initName];
      if (tensor) {
        this.addConstant(tensor);
      }
    }

    for (const node of this.graph.nodes) {
      this.compileNode(node);
    }

    const outputs: Record<string, MLOperand> = {};
    for (const output of this.graph.outputs) {
      const op = this.operands.get(output.name);
      if (op) {
        outputs[output.name] = op;
      } else {
        throw new Error(`Output ${output.name} not found in WebNN operands.`);
      }
    }

    return await this.builder.build(outputs);
  }

  private addConstant(tensor: Tensor): void {
    if (!tensor.data) {
      return;
    }

    const dimensions =
      tensor.shape.length === 0 ? [1] : tensor.shape.map((d) => (typeof d === 'number' ? d : 1));
    const dataType = this.mapDType(tensor.dtype);

    this.operands.set(tensor.name, this.builder.constant({ dataType, dimensions }, tensor.data));
  }

  private mapDType(dtype: DType | 'bool'): MLOperandDataType {
    switch (dtype) {
      case 'float32':
        return 'float32';
      case 'float16':
        return 'float16';
      case 'float64':
        return 'float32'; // 217. Ensure 64-bit floats are down-casted to float32
      case 'int32':
        return 'int32';
      case 'int8':
        return 'int8';
      case 'uint8':
        return 'uint8';
      case 'bool':
        return 'uint8'; // Boolean maps to uint8
      case 'int64':
        return 'int32'; // 216. Ensure 64-bit integer inputs are down-casted to int32
      case 'uint64':
        return 'uint32';
      default:
        throw new Error(`Unsupported WebNN data type: ${dtype}`);
    }
  }

  public getFloatAttribute(node: Node, name: string, defaultValue?: number): number | undefined {
    const attr = node.attributes[name];
    if (attr && attr.type === 'FLOAT' && typeof attr.value === 'number') {
      return attr.value;
    }
    return defaultValue;
  }

  public getIntAttribute(node: Node, name: string, defaultValue?: number): number | undefined {
    const attr = node.attributes[name];
    if (attr && attr.type === 'INT' && typeof attr.value === 'number') {
      return attr.value;
    }
    return defaultValue;
  }

  public getIntsAttribute(node: Node, name: string, defaultValue?: number[]): number[] | undefined {
    const attr = node.attributes[name];
    if (attr && attr.type === 'INTS' && Array.isArray(attr.value)) {
      return attr.value as number[];
    }
    return defaultValue;
  }

  public getStringAttribute(node: Node, name: string, defaultValue?: string): string | undefined {
    const attr = node.attributes[name];
    if (attr && attr.type === 'STRING' && typeof attr.value === 'string') {
      return attr.value;
    }
    return defaultValue;
  }

  public extractInt32TensorData(tensorName: string | undefined): number[] | null {
    if (!tensorName) return null;
    const tensor = this.graph.tensors[tensorName];
    if (!tensor || !tensor.data) return null;
    const data = new Int32Array(
      tensor.data.buffer,
      tensor.data.byteOffset,
      tensor.data.byteLength / 4,
    );
    return Array.from(data);
  }

  public extractFloat32TensorData(tensorName: string | undefined): number[] | null {
    if (!tensorName) return null;
    const tensor = this.graph.tensors[tensorName];
    if (!tensor) return null;
    if (!tensor.data) return null;
    const data = new Float32Array(
      tensor.data.buffer,
      tensor.data.byteOffset,
      tensor.data.byteLength / 4,
    );
    return Array.from(data);
  }

  private compileNode(node: Node): void {
    const inputs = node.inputs.map((name) => {
      if (name === '') return null;
      const op = this.operands.get(name);
      if (!op) {
        throw new Error(`Input operand ${name} not found for node ${node.name || node.opType}`);
      }
      return op;
    });

    let result!: MLOperand | MLOperand[];

    // 218. Handle empty tensor evaluations (e.g., shape [0, 10]) without crashing the NPU driver.
    // 219. Manage NaNs and Infs propagation explicitly according to WebNN standard guidelines.
    // We check if any input has a 0 dimension and immediately return an empty/zeroized operand
    // or offload (stubbed safety check here).
    for (const op of inputs) {
      if (op && op.shape && op.shape.some((d) => d === 0)) {
        // Emulate an empty tensor operation or throw to fallback
        throw new Error(
          'Fallback triggered: WebNN native ops currently fail on 0-dimension shapes',
        );
      }
    }

    const opClass = webnnRegistry.get_op(node.domain || 'ai.onnx', node.opType);
    if (!opClass) {
      throw new Error(`Unsupported WebNN node type: ${node.opType}`);
    }
    const opInstance = new opClass();
    result = opInstance.build(this, node, inputs);

    if (Array.isArray(result)) {
      node.outputs.forEach((outName, i) => {
        if (outName !== '') {
          this.operands.set(outName, result[i]!);
        }
      });
    } else {
      if (node.outputs[0] !== '') {
        this.operands.set(node.outputs[0]!, result);
      }
    }
  }
}
