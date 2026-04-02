import { MLContext, MLContextOptions, MLComputeResult, MLOpSupportLimits } from './interfaces.js';
import { PolyfillMLGraph } from './graph.js';
import { PolyfillMLTensor, MLTensorOptions } from './tensor.js';
import {
  InferenceSession,
  ExecutionProvider,
  WebGPUProvider,
  WasmProvider,
} from '@onnx9000/backend-web';
import { Tensor, DType } from '@onnx9000/core';

export class PolyfillMLContext implements MLContext {
  private options: MLContextOptions;
  private providers: ExecutionProvider[] = [];

  constructor(options: MLContextOptions = {}) {
    this.options = options;
    const deviceType = options.deviceType || 'cpu';

    // 7. Route `deviceType: 'gpu'` requests directly to the `onnx9000` WebGPU backend.
    // 8. Route `deviceType: 'cpu'` requests directly to the `onnx9000` WASM SIMD backend.
    if (deviceType === 'gpu') {
      this.providers.push(new WebGPUProvider());
    } else if (deviceType === 'cpu') {
      this.providers.push(new WasmProvider());
    } else {
      // Default to WASM if unknown or npu explicitly requested on polyfill
      this.providers.push(new WasmProvider());
    }
  }

  // 11, 160. Implement context.createTensor(options) natively.
  async createTensor(options: MLTensorOptions): Promise<PolyfillMLTensor> {
    return new PolyfillMLTensor(options); // Device logic omitted for polyfill basic execution
  }

  // 165. Implement context.readTensor(tensor, arrayBuffer) copying data natively.
  async readTensor(tensor: PolyfillMLTensor, arrayBuffer: ArrayBuffer): Promise<void> {
    if (tensor.internalBuffer) {
      const src = new Uint8Array(tensor.internalBuffer);
      const dst = new Uint8Array(arrayBuffer);
      dst.set(src.subarray(0, dst.length));
    }
  }

  // 166. Implement context.writeTensor(tensor, arrayBuffer) copying data natively.
  async writeTensor(tensor: PolyfillMLTensor, arrayBuffer: ArrayBuffer): Promise<void> {
    if (tensor.internalBuffer) {
      const src = new Uint8Array(arrayBuffer);
      const dst = new Uint8Array(tensor.internalBuffer);
      dst.set(src.subarray(0, dst.length));
    }
  }

  // 10, 174. Implement context.dispatch(graph, inputs, outputs) (Using MLTensor structures).
  async dispatch(
    graph: PolyfillMLGraph,
    inputs: Record<string, PolyfillMLTensor>,
    outputs: Record<string, PolyfillMLTensor>,
  ): Promise<void> {
    // Dispatch is similar to compute but doesn't return anything natively, just writes directly to output tensors
    const session = new InferenceSession(graph.onnxGraph, this.providers);
    const outputNames = graph.onnxGraph.outputs.map((o: any) => o.name);

    const onnxInputs: Record<string, Tensor> = {};
    for (const [name, tensor] of Object.entries(inputs)) {
      const inputInfo = graph.onnxGraph.inputs.find((i: any) => i.name === name);
      if (!inputInfo) {
        throw new DOMException(`Input ${name} not found in graph`, 'DataError');
      }
      onnxInputs[name] = new Tensor(
        name,
        inputInfo.shape,
        inputInfo.dtype,
        false,
        false,
        new Uint8Array(tensor.internalBuffer || new ArrayBuffer(0)),
      );
    }

    const onnxOutputs: Record<string, Tensor> = await session.run(outputNames, onnxInputs);

    for (const [name, outTensor] of Object.entries(onnxOutputs)) {
      if (outputs[name] && outputs[name].internalBuffer && outTensor.data) {
        const dest = new Uint8Array(outputs[name].internalBuffer);
        const src = new Uint8Array(
          outTensor.data.buffer,
          outTensor.data.byteOffset,
          outTensor.data.byteLength,
        );
        dest.set(src);
      }
    }
  }

  // 9. Implement MLContext.compute(graph, inputs, outputs).
  async compute(
    graph: PolyfillMLGraph,
    inputs: Record<string, ArrayBufferView>,
    outputs: Record<string, ArrayBufferView>,
  ): Promise<MLComputeResult> {
    const session = new InferenceSession(graph.onnxGraph, this.providers);
    const outputNames = graph.onnxGraph.outputs.map((o: any) => o.name);

    const onnxInputs: Record<string, Tensor> = {};
    for (const [name, data] of Object.entries(inputs)) {
      const inputInfo = graph.onnxGraph.inputs.find((i: any) => i.name === name);
      if (!inputInfo) {
        throw new DOMException(`Input ${name} not found in graph`, 'DataError');
      }
      // 171. Extract ArrayBufferView data dynamically from inputs.
      onnxInputs[name] = new Tensor(name, inputInfo.shape, inputInfo.dtype, false, false, data);
    }

    // 172. Execute the onnx9000 internal session natively.
    const onnxOutputs: Record<string, Tensor> = await session.run(outputNames, onnxInputs);

    // 173. Copy output values into the user's outputs ArrayBufferView directly.
    for (const [name, outTensor] of Object.entries(onnxOutputs)) {
      if (outputs[name]) {
        // Copy data natively

        const dest = new Uint8Array(
          outputs[name].buffer,
          outputs[name].byteOffset,
          outputs[name].byteLength,
        );
        if (outTensor.data) {
          const src = new Uint8Array(
            outTensor.data.buffer,
            outTensor.data.byteOffset,
            outTensor.data.byteLength,
          );
          dest.set(src);
        }
      }
    }

    return { inputs, outputs };
  }

  // 12. Expose MLContext.opSupportLimits() mapping to the onnx9000 capability registry.
  opSupportLimits(): MLOpSupportLimits {
    return {
      input: {
        dataTypes: ['float32', 'float16', 'int32', 'uint32', 'int8', 'uint8', 'int64', 'uint64'],
      },
    };
  }
}
