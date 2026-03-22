import { DType, Shape } from '@onnx9000/core';
import { TensorType } from '../flatbuffer/schema';

export function mapOnnxTypeToTflite(dtype: DType, name?: string): TensorType {
  switch (dtype) {
    // 55. Map ONNX FLOAT -> TFLite FLOAT32.
    case 'float32':
      return TensorType.FLOAT32;
    // 56. Map ONNX FLOAT16 -> TFLite FLOAT16.
    case 'float16':
      return TensorType.FLOAT16;
    // 57. Map ONNX INT32 -> TFLite INT32.
    case 'int32':
      return TensorType.INT32;
    // 58. Map ONNX INT64 -> TFLite INT64.
    // 314. Prevent Int64 tensor generation inside mobile targets (converting natively to Int32 and warning user).
    case 'int64':
      if (name) {
        console.warn(
          `[onnx2tf] Warning: Downcasting Int64 tensor '${name}' to Int32 for mobile compatibility.`,
        );
      }
      return TensorType.INT32;
    // 59. Map ONNX INT8 -> TFLite INT8.
    case 'int8':
      return TensorType.INT8;
    // 60. Map ONNX UINT8 -> TFLite UINT8.
    case 'uint8':
      return TensorType.UINT8;
    // 61. Map ONNX BOOL -> TFLite BOOL.
    case 'bool':
      return TensorType.BOOL;
    // 62. Map ONNX STRING -> TFLite STRING.
    case 'string':
      return TensorType.STRING;
    // 63. Handle ONNX DOUBLE (Float64) gracefully (downcast to Float32).
    case 'float64':
      return TensorType.FLOAT32;
    default:
      // 73. Provide fallback casting if TFLite lacks an op signature for a specific type.
      return TensorType.FLOAT32;
  }
}

export function mapOnnxShapeToTflite(shape: Shape): number[] {
  // 64. Map empty ONNX shapes [] to TFLite scalar shapes [].
  if (!shape || shape.length === 0) {
    return [];
  }

  return shape.map((dim) => {
    if (typeof dim === 'number') {
      // 65. Map dynamic ONNX shapes [-1, 224, 224, 3] safely.
      return dim >= 0 ? dim : -1;
    }
    // Dynamic string dimensions become -1 in TFLite
    return -1;
  });
}

export function createShapeSignature(shape: Shape): number[] {
  // 66. Emit ShapeSignature vectors for TFLite dynamic shapes.
  return mapOnnxShapeToTflite(shape);
}
