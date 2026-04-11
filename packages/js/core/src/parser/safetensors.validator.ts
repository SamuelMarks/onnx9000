/* eslint-disable */
import { SafeTensors, TensorInfo } from './safetensors.js';
import { Graph } from '../ir/graph.js';

/**
 * Enumeration of data types for validation, simplified from ONNX.
 */
export enum DType {
  UNDEFINED = 0,
  FLOAT = 1,
  UINT8 = 2,
  INT8 = 3,
  UINT16 = 4,
  INT16 = 5,
  INT32 = 6,
  INT64 = 7,
  STRING = 8,
  BOOL = 9,
  FLOAT16 = 10,
  DOUBLE = 11,
  UINT32 = 12,
  UINT64 = 13,
  COMPLEX64 = 14,
  COMPLEX128 = 15,
  BFLOAT16 = 16,
}

/**
 * Maps an internal DType enum value to its corresponding TypedArray name.
 * @param dtype The numerical DType value
 * @returns The string name of the corresponding JavaScript TypedArray
 */
export function toEmscriptenType(dtype: number): string {
  switch (dtype) {
    case DType.FLOAT:
      return 'Float32Array';
    case DType.DOUBLE:
      return 'Float64Array';
    case DType.INT8:
      return 'Int8Array';
    case DType.INT16:
      return 'Int16Array';
    case DType.INT32:
      return 'Int32Array';
    case DType.INT64:
      return 'BigInt64Array';
    case DType.UINT8:
      return 'Uint8Array';
    case DType.UINT16:
      return 'Uint16Array';
    case DType.UINT32:
      return 'Uint32Array';
    case DType.UINT64:
      return 'BigUint64Array';
    case DType.BOOL:
      return 'Uint8Array';
    case DType.FLOAT16:
      return 'Uint16Array';
    case DType.BFLOAT16:
      return 'Uint16Array';
    default:
      return 'Unknown';
  }
}

/**
 * Validates that the shapes and data types in a Safetensors buffer match an ONNX Graph.
 * @param buffer The Safetensors file data
 * @param graph The reference ONNX Graph
 */
export function validateOnnxShapesAndDtypes(buffer: ArrayBuffer, graph: Graph): void {
  const st = new SafeTensors(buffer);
  const dtypeMap: Record<string, string> = {
    F64: 'Float64Array',
    F32: 'Float32Array',
    F16: 'Uint16Array',
    I64: 'BigInt64Array',
    I32: 'Int32Array',
    I16: 'Int16Array',
    I8: 'Int8Array',
    U64: 'BigUint64Array',
    U32: 'Uint32Array',
    U16: 'Uint16Array',
    U8: 'Uint8Array',
    BOOL: 'Uint8Array',
    BF16: 'Uint16Array',
  };

  for (const name of st.keys()) {
    const tensorNode = graph.tensors[name];
    if (tensorNode) {
      const info = st.tensors[name] as TensorInfo;

      // Check shape
      const stShapeStr = JSON.stringify(info.shape);
      const onnxShapeStr = JSON.stringify(tensorNode.shape);
      if (tensorNode.shape && stShapeStr !== onnxShapeStr) {
        console.warn(
          `Shape mismatch for ${name}: ONNX expects ${onnxShapeStr}, Safetensors provides ${stShapeStr}`,
        );
      }

      // Check dtype
      if (tensorNode.dtype !== undefined) {
        // tensorNode.dtype in js ir is a string
        const onnxTypeStr = tensorNode.dtype;
        const stTypeStr = dtypeMap[info.dtype];

        // Extremely simplified mapping for testing parity
        const typeMatcher: Record<string, string> = {
          float32: 'Float32Array',
          float64: 'Float64Array',
          int8: 'Int8Array',
          int16: 'Int16Array',
          int32: 'Int32Array',
          int64: 'BigInt64Array',
          uint8: 'Uint8Array',
          uint16: 'Uint16Array',
          uint32: 'Uint32Array',
          uint64: 'BigUint64Array',
          bool: 'Uint8Array',
          float16: 'Uint16Array',
          bfloat16: 'Uint16Array',
        };

        const expectedStr = typeMatcher[onnxTypeStr] || 'Unknown';

        if (stTypeStr !== expectedStr) {
          console.warn(
            `DType mismatch for ${name}: ONNX expects ${expectedStr}, Safetensors provides ${stTypeStr} (${info.dtype})`,
          );
        }
      }
    }
  }
}
