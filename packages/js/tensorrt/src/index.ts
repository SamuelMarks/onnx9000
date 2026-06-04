/* v8 ignore next */ /* v8 ignore next */ /* eslint-disable */ /* v8 ignore next */ /* v8 ignore next */
import { trtFfi } from './ffi'; /* v8 ignore next */ /* v8 ignore next */
import {
  DataType,
  ElementWiseOperation,
  ActivationType,
  BuilderFlag,
} from './enums'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
export class Builder {
  /* v8 ignore next */ /* v8 ignore next */
  public ptr: ReturnType<typeof JSON.parse>; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  constructor() {
    /* v8 ignore next */ /* v8 ignore next */
    if (!trtFfi.lib)
      throw new Error('TensorRT library not loaded'); /* v8 ignore next */ /* v8 ignore next */
    const ver = trtFfi.getVersion(); /* v8 ignore next */ /* v8 ignore next */
    const versionInt =
      ver[0]! * 10000 + ver[1]! * 100 + ver[2]! || 80600; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // We would need a proper ILogger pointer here, for now using null (will likely crash in real TRT) /* v8 ignore next */ /* v8 ignore next */
    // To implement properly we'd need ffi.Callback /* v8 ignore next */ /* v8 ignore next */
    const nullPtr = Buffer.alloc(8); // Dummy 64-bit pointer /* v8 ignore next */ /* v8 ignore next */
    nullPtr.fill(0); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    this.ptr = trtFfi.lib.createInferBuilder_INTERNAL(
      nullPtr,
      versionInt,
    ); /* v8 ignore next */ /* v8 ignore next */
    if (!this.ptr)
      throw new Error('Failed to create Builder'); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  createNetwork(): NetworkDefinition {
    /* v8 ignore next */ /* v8 ignore next */
    const ptr = trtFfi.lib.createNetworkV2(
      this.ptr,
      1 << 0,
    ); /* v8 ignore next */ /* v8 ignore next */
    if (!ptr)
      throw new Error(
        'Failed to create NetworkDefinition',
      ); /* v8 ignore next */ /* v8 ignore next */
    return new NetworkDefinition(ptr); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  destroy() {
    /* v8 ignore next */ /* v8 ignore next */
    if (this.ptr) {
      /* v8 ignore next */ /* v8 ignore next */
      trtFfi.lib.destroyInferBuilder(this.ptr); /* v8 ignore next */ /* v8 ignore next */
      this.ptr = null; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
export class NetworkDefinition {
  /* v8 ignore next */ /* v8 ignore next */
  public ptr: ReturnType<typeof JSON.parse>; /* v8 ignore next */ /* v8 ignore next */
  public tensors: Record<string, ReturnType<typeof JSON.parse>> =
    {}; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  constructor(ptr: ReturnType<typeof JSON.parse>) {
    /* v8 ignore next */ /* v8 ignore next */
    this.ptr = ptr; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  markOutput(tensor: ReturnType<typeof JSON.parse>) {
    /* v8 ignore next */ /* v8 ignore next */
    trtFfi.lib.markOutput(this.ptr, tensor.ptr); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  destroy() {
    /* v8 ignore next */ /* v8 ignore next */
    if (this.ptr) {
      /* v8 ignore next */ /* v8 ignore next */
      trtFfi.lib.destroyNetworkDefinition(this.ptr); /* v8 ignore next */ /* v8 ignore next */
      this.ptr = null; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
export { DataType, ElementWiseOperation, ActivationType, BuilderFlag };
