import { MLOperandDataType } from './interfaces.js';

export interface MLTensorOptions {
  dataType: MLOperandDataType;
  dimensions: number[];
  usage?: number; // MLTensorUsage
}

export class PolyfillMLTensor {
  dataType: MLOperandDataType;
  dimensions: number[];
  usage: number;
  buffer: any | null = null;
  internalBuffer: ArrayBuffer | null = null; // fallback for WASM

  constructor(options: MLTensorOptions, device?: any) {
    this.dataType = options.dataType;
    this.dimensions = options.dimensions;
    this.usage = options.usage || 0;

    // Simulate mapping to an internal WebGPU buffer (161)
    if (device) {
      const size = this.calculateSize(this.dataType, this.dimensions);
      this.buffer = device.createBuffer({
        size,
        usage: this.usage,
      });
    } else {
      const size = this.calculateSize(this.dataType, this.dimensions);
      this.internalBuffer = new ArrayBuffer(size);
    }
  }

  private calculateSize(dataType: MLOperandDataType, dimensions: number[]): number {
    const counts = dimensions.reduce((a, b) => a * b, 1);
    const byteMap: Record<string, number> = {
      float32: 4,
      float16: 2,
      int32: 4,
      uint32: 4,
      int8: 1,
      uint8: 1,
      int64: 8,
      uint64: 8,
    };
    return counts * (byteMap[dataType] || 4);
  }

  // 167. Implement MLTensor.destroy() hooking directly into buffer.destroy().
  destroy() {
    if (this.buffer) {
      this.buffer.destroy();
      this.buffer = null;
    }
    this.internalBuffer = null;
  }
}
