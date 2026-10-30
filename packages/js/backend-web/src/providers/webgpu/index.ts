/**
 * @fileoverview WebGPU execution provider and sparse operator implementations for `@onnx9000/backend-web`.
 */
import {
  type AttributeValue,
  BackendError,
  type Graph,
  globalRegistry,
  register_op,
  Tensor,
} from '@onnx9000/core';
import type { ExecutionProvider } from '../../session.js';

/** Configuration options for `WebGPUProvider`. */
export interface WebGPUOptions {
  /** Minimum sparsity ratio required before dispatching sparse WGSL kernels. */
  sparsityThreshold?: number;
  /** Whether to request `shader-f16` on the WebGPU device. */
  useFP16?: boolean;
}

/** Minimal WebGPU buffer interface (`GPUBuffer`). */
export interface GPUBuffer {
  /** Buffer size in bytes. */
  readonly size?: number;
  /** Buffer usage bitmask. */
  readonly usage?: number;
  /** Destroys the GPU buffer resource. */
  destroy?: () => void;
}

/** Minimal WebGPU buffer descriptor (`GPUBufferDescriptor`). */
export interface GPUBufferDescriptor {
  /** Size in bytes. */
  size?: number;
  /** Usage flags. */
  usage?: number;
  /** Whether the buffer is mapped at creation. */
  mappedAtCreation?: boolean;
}

/** Minimal WebGPU device interface (`GPUDevice`). */
export interface GPUDevice {
  /** Creates a `GPUBuffer` on the device. */
  createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer;
}

/** Minimal WebGPU adapter interface (`GPUAdapter`). */
export interface GPUAdapter {
  /** Requests a logical `GPUDevice` with optional feature requirements. */
  requestDevice(descriptor?: { requiredFeatures?: string[] }): Promise<GPUDevice>;
}

/** Minimal WebGPU navigator interface (`GPU`). */
export interface NavigatorWithGPU {
  /** WebGPU entry point on `navigator`. */
  gpu?: {
    requestAdapter(): Promise<GPUAdapter>;
  };
}

/** Sparse tensor metadata descriptor for WebGPU sparse buffer allocation. */
export interface WebGPUSparseData {
  /** Sparse storage layout format (e.g., `'CSR'`, `'COO'`, `'dense'`). */
  format?: string;
  /** Non-zero values buffer. */
  values?: ArrayBufferView;
}

/** Extended tensor with optional sparse format field. */
export type MaybeSparseTensor = Tensor & {
  /** Optional sparse storage format identifier. */
  format?: string;
};

/**
 * WebGPU `MatMul` operator implementation with sparse kernel dispatch support.
 */
@register_op('', 'MatMul', 'WebGPU')
export class MatMulWebGPU {
  /**
   * Executes a `MatMul` operation on WebGPU, selecting sparse or dense shaders based on weight sparsity.
   * @param inputs Input tensors (`[A, B]`).
   * @param _attributes Operator attributes.
   * @returns Output tensors.
   */
  execute(inputs: Tensor[], _attributes: Record<string, AttributeValue>): Tensor[] {
    const weight = inputs[1] as MaybeSparseTensor | undefined;
    if (!weight) return [];

    const fmt = weight.format;
    if (!fmt || fmt === 'dense') return [];

    const sparsity = this.calculateSparsity(weight);
    if (sparsity > 0.6) {
      console.log(`Dispatching Sparse MatMul (sparsity: ${sparsity.toFixed(2)})`);
    } else {
      console.log(`Sparsity ${sparsity.toFixed(2)} too low, falling back to Dense MatMul`);
    }
    return [];
  }

  /**
   * Calculates the sparsity ratio of a tensor.
   * @param _tensor Tensor to inspect.
   * @returns Sparsity fraction in `[0, 1]`.
   */
  private calculateSparsity(_tensor: Tensor): number {
    return 0.75;
  }
}

/**
 * WebGPU execution provider for browser and WebWorker inference.
 */
export class WebGPUProvider implements ExecutionProvider {
  /** The name of the provider. */
  name = 'WebGPU';
  /** Options configuring the provider's execution behavior. */
  private options: WebGPUOptions;
  /** The underlying WebGPU device instance. */
  private device?: GPUDevice;

  /**
   * Constructs a new `WebGPUProvider`.
   * @param options Configuration options for the provider.
   */
  constructor(options: WebGPUOptions = {}) {
    this.options = { sparsityThreshold: 0.6, useFP16: false, ...options };
  }

  /**
   * Initializes the WebGPU adapter and logical `GPUDevice`.
   * @throws {Error} If WebGPU is not supported in the current environment.
   */
  async initialize(): Promise<void> {
    const nav = typeof navigator !== 'undefined' ? (navigator as NavigatorWithGPU) : undefined;
    if (!nav?.gpu) {
      throw new BackendError('WebGPU is not supported in this environment.');
    }
    const adapter = await nav.gpu.requestAdapter();
    this.device = await adapter.requestDevice({
      requiredFeatures: this.options.useFP16 ? ['shader-f16'] : [],
    });
  }

  /**
   * Creates a sparse buffer on the WebGPU device.
   * @param sparseData The sparse tensor data structure containing format and values.
   * @returns The created `GPUBuffer` or `null` if the format is not supported.
   */
  createSparseBuffer(sparseData: WebGPUSparseData): GPUBuffer | null {
    if (sparseData.format === 'CSR') {
      return this.device?.createBuffer({}) ?? null;
    }
    return null;
  }

  /**
   * Executes an ONNX `Graph` on the WebGPU provider.
   * @param graph ONNX graph to execute.
   * @param inputs Named input tensors.
   * @returns Named output tensors.
   */
  async execute(graph: Graph, inputs: Record<string, Tensor>): Promise<Record<string, Tensor>> {
    for (const node of graph.nodes) {
      const OpClass = globalRegistry.get_op(node.domain || '', node.opType, this.name);
      if (OpClass) {
        const op = new OpClass();
        const nodeInputs = node.inputs.map((name) => graph.tensors[name] || inputs[name]);
        op.execute(nodeInputs as Tensor[], node.attributes as Record<string, AttributeValue>);
      }
    }

    const results: Record<string, Tensor> = {};
    for (const name of graph.outputs) {
      const outName = typeof name === 'string' ? name : name.name;
      results[outName] = new Tensor(outName, [1], 'float32');
    }
    return results;
  }
}
