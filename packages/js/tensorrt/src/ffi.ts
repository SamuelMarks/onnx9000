import ffi from 'ffi-napi';
import ref from 'ref-napi';
import * as os from 'os';

export class TensorRTFFI {
  public lib: any;

  constructor() {
    this.loadLibrary();
  }

  private loadLibrary() {
    const isWindows = os.platform() === 'win32';
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
      });
    } catch (e) {
      console.warn(`Could not load TensorRT library: ${e}`);
    }
  }

  public getVersion(): number[] {
    if (!this.lib || !this.lib.getInferLibVersion) return [0, 0, 0];
    const ver = this.lib.getInferLibVersion();
    const major = Math.floor(ver / 1000);
    const minor = Math.floor((ver % 1000) / 100);
    const patch = ver % 100;
    return [major, minor, patch];
  }
}

export const trtFfi = new TensorRTFFI();
