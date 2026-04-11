/* eslint-disable */
import { trtFfi } from './ffi';
import { DataType, ElementWiseOperation, ActivationType, BuilderFlag } from './enums';

export class Builder {
  public ptr: ReturnType<typeof JSON.parse>;

  constructor() {
    if (!trtFfi.lib) throw new Error('TensorRT library not loaded');
    const ver = trtFfi.getVersion();
    const versionInt = ver[0]! * 10000 + ver[1]! * 100 + ver[2]! || 80600;

    // We would need a proper ILogger pointer here, for now using null (will likely crash in real TRT)
    // To implement properly we'd need ffi.Callback
    const nullPtr = Buffer.alloc(8); // Dummy 64-bit pointer
    nullPtr.fill(0);

    this.ptr = trtFfi.lib.createInferBuilder_INTERNAL(nullPtr, versionInt);
    if (!this.ptr) throw new Error('Failed to create Builder');
  }

  createNetwork(): NetworkDefinition {
    const ptr = trtFfi.lib.createNetworkV2(this.ptr, 1 << 0);
    if (!ptr) throw new Error('Failed to create NetworkDefinition');
    return new NetworkDefinition(ptr);
  }

  destroy() {
    if (this.ptr) {
      trtFfi.lib.destroyInferBuilder(this.ptr);
      this.ptr = null;
    }
  }
}

export class NetworkDefinition {
  public ptr: ReturnType<typeof JSON.parse>;
  public tensors: Record<string, ReturnType<typeof JSON.parse>> = {};

  constructor(ptr: ReturnType<typeof JSON.parse>) {
    this.ptr = ptr;
  }

  markOutput(tensor: ReturnType<typeof JSON.parse>) {
    trtFfi.lib.markOutput(this.ptr, tensor.ptr);
  }

  destroy() {
    if (this.ptr) {
      trtFfi.lib.destroyNetworkDefinition(this.ptr);
      this.ptr = null;
    }
  }
}

export { DataType, ElementWiseOperation, ActivationType, BuilderFlag };
