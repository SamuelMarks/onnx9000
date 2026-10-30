/**
 * @fileoverview ffi-napi.d.ts
 * Type declarations for the native `ffi-napi` library used by the TensorRT FFI bindings.
 */
declare module 'ffi-napi' {
  /** Native pointer handle represented as a Node.js Buffer or opaque object. */
  export type NativePointer = Buffer | Record<string, unknown> | null;

  /** Loaded TensorRT native library interface. */
  export interface TensorRTNativeLibrary {
    getInferLibVersion?: () => number;
    createInferBuilder_INTERNAL: (logger: NativePointer, version: number) => NativePointer;
    createNetworkV2: (builder: NativePointer, flags: number) => NativePointer;
    destroyInferBuilder: (builder: NativePointer) => void;
    destroyNetworkDefinition: (network: NativePointer) => void;
    addInput?: (
      network: NativePointer,
      name: string,
      dtype: number,
      dims: NativePointer,
    ) => NativePointer;
    markOutput: (network: NativePointer, tensor: NativePointer) => void;
    addElementWise?: (
      network: NativePointer,
      input1: NativePointer,
      input2: NativePointer,
      op: number,
    ) => NativePointer;
    addActivation?: (network: NativePointer, input: NativePointer, type: number) => NativePointer;
    addMatrixMultiply?: (
      network: NativePointer,
      input0: NativePointer,
      op0: number,
      input1: NativePointer,
      op1: number,
    ) => NativePointer;
    addPoolingNd?: (
      network: NativePointer,
      input: NativePointer,
      type: number,
      windowSize: NativePointer,
    ) => NativePointer;
  }

  /** FFI signature definition mapping method names to [returnType, argTypes]. */
  export type FFILibrarySpec = Record<string, [string, string[]]>;

  /** FFI module namespace interface. */
  export interface FFIModule {
    Library: (libFile: string, funcs: FFILibrarySpec) => TensorRTNativeLibrary;
  }

  const ffi: FFIModule;
  export default ffi;
}
