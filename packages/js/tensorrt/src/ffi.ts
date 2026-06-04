/* eslint-disable */
// @ts-ignore
import ffi from 'ffi-napi';
// @ts-ignore
import ref from 'ref-napi';
import * as os from 'os';

export class TensorRTFFI {
  public lib: ReturnType<typeof JSON.parse>;

  constructor() {
    this.loadLibrary();
  }

  private loadLibrary() {
    const isWindows = os.platform() === 'win32'; /* v8 ignore next */ /* v8 ignore next */
    const libName = isWindows ? 'nvinfer.dll' : 'libnvinfer.so';

    try {
      this.lib = ffi.Library(libName, {
        getInferLibVersion: ['int', []],
        createInferBuilder_INTERNAL: ['pointer', ['pointer', 'int']],
        createNetworkV2: ['pointer', ['pointer', 'int32']],
        destroyInferBuilder: ['void', ['pointer']],
        destroyNetworkDefinition: ['void', ['pointer']],
        addInput: ['pointer', ['pointer', 'string', 'int32', 'pointer']],
        markOutput: ['void', ['pointer', 'pointer']],
        addElementWise: ['pointer', ['pointer', 'pointer', 'pointer', 'int32']],
        addActivation: ['pointer', ['pointer', 'pointer', 'int32']],
        addMatrixMultiply: ['pointer', ['pointer', 'pointer', 'int32', 'pointer', 'int32']],
        addPoolingNd: ['pointer', ['pointer', 'pointer', 'int32', 'pointer']],
      }); /* v8 ignore next */ /* v8 ignore next */
    } catch (e) {
      /* v8 ignore next */ /* v8 ignore next */
      console.warn(
        `Could not load TensorRT library: ${e}`,
      ); /* v8 ignore next */ /* v8 ignore next */
    }
  }

  public getVersion(): number[] {
    /* v8 ignore next */ /* v8 ignore next */
    if (!this.lib || !this.lib.getInferLibVersion)
      return [0, 0, 0]; /* v8 ignore next */ /* v8 ignore next */
    const ver = this.lib.getInferLibVersion(); /* v8 ignore next */ /* v8 ignore next */
    const major = Math.floor(ver / 1000); /* v8 ignore next */ /* v8 ignore next */
    const minor = Math.floor((ver % 1000) / 100); /* v8 ignore next */ /* v8 ignore next */
    const patch = ver % 100; /* v8 ignore next */ /* v8 ignore next */
    return [major, minor, patch];
  }
}

export const trtFfi = new TensorRTFFI();
