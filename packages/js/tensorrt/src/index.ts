/**
 * @fileoverview index.ts
 * Provides index functionality for the tensorrt package.
 */

import { BackendError } from '@onnx9000/core';
import type { NativePointer } from 'ffi-napi';
import { ActivationType, BuilderFlag, DataType, ElementWiseOperation } from './enums';
import { trtFfi } from './ffi';

/** Represents the `Builder` class. */
export class Builder {
  /** Property `ptr` of `Builder`. */
  public ptr: NativePointer;
  private lib: NonNullable<typeof trtFfi.lib>;

  /** Initializes a new `Builder` instance. */
  constructor() {
    if (!trtFfi.lib) throw new BackendError('TensorRT library not loaded');
    this.lib = trtFfi.lib;
    const ver = trtFfi.getVersion();
    const versionInt =
      (ver[0] as number) * 10000 + (ver[1] as number) * 100 + (ver[2] as number) || 80600;

    // We would need a proper ILogger pointer here, for now using null (will likely crash in real TRT)
    // To implement properly we'd need ffi.Callback
    const nullPtr = Buffer.alloc(8); // Dummy 64-bit pointer
    nullPtr.fill(0);

    this.ptr = this.lib.createInferBuilder_INTERNAL(nullPtr, versionInt);
    if (!this.ptr) throw new BackendError('Failed to create Builder');
  }

  /**
   * Executes `createNetwork` on `Builder`.
   * @returns Result of `createNetwork`.
   */
  createNetwork(): NetworkDefinition {
    const ptr = this.lib.createNetworkV2(this.ptr, 1 << 0);
    if (!ptr) throw new BackendError('Failed to create NetworkDefinition');
    return new NetworkDefinition(ptr, this.lib);
  }

  /**
   * Executes `destroy` on `Builder`.
   * @returns Result of `destroy`.
   */
  destroy() {
    if (this.ptr) {
      this.lib.destroyInferBuilder(this.ptr);
      this.ptr = null;
    }
  }
}

/** Represents a native TensorRT tensor handle. */
export interface TensorHandle {
  /** Native pointer to the underlying ITensor. */
  ptr: NativePointer;
}

/** Represents the `NetworkDefinition` class. */
export class NetworkDefinition {
  /** Property `ptr` of `NetworkDefinition`. */
  public ptr: NativePointer;
  /** Property `tensors` of `NetworkDefinition`. */
  public tensors: Record<string, TensorHandle> = {};
  private lib: NonNullable<typeof trtFfi.lib>;

  /**
   * Initializes a new `NetworkDefinition` instance.
   * @param ptr Parameter `ptr`.
   * @param lib Native TensorRT FFI library reference.
   */
  constructor(ptr: NativePointer, lib: NonNullable<typeof trtFfi.lib>) {
    this.ptr = ptr;
    this.lib = lib;
  }

  /**
   * Executes `markOutput` on `NetworkDefinition`.
   * @param tensor Parameter `tensor`.
   * @returns Result of `markOutput`.
   */
  markOutput(tensor: TensorHandle) {
    this.lib.markOutput(this.ptr, tensor.ptr);
  }

  /**
   * Executes `destroy` on `NetworkDefinition`.
   * @returns Result of `destroy`.
   */
  destroy() {
    if (this.ptr) {
      this.lib.destroyNetworkDefinition(this.ptr);
      this.ptr = null;
    }
  }
}

/** Represents the `TensorRTProvider` class. */
export class TensorRTProvider {
  /** Property `graph` of `TensorRTProvider`. */
  public graph: unknown;

  /**
   * Initializes a new `TensorRTProvider` instance.
   * @param graph Parameter `graph`.
   */
  constructor(graph?: unknown) {
    this.graph = graph;
  }
}

export { ActivationType, BuilderFlag, DataType, ElementWiseOperation };
