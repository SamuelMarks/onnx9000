import { describe, expect, it } from 'vitest';
import {
  BackendError,
  ConversionError,
  ONNXParseError,
  Onnx9000Error,
  ShapeInferenceError,
  UnsupportedOpError,
} from '../src/exceptions.js';

describe('Onnx9000Error hierarchy', () => {
  it('constructs base Onnx9000Error with default and custom messages', () => {
    const defaultErr = new Onnx9000Error();
    expect(defaultErr).toBeInstanceOf(Error);
    expect(defaultErr).toBeInstanceOf(Onnx9000Error);
    expect(defaultErr.message).toBe('An error occurred');
    expect(defaultErr.name).toBe('Onnx9000Error');

    const customErr = new Onnx9000Error('custom failure');
    expect(customErr.message).toBe('custom failure');
  });

  it('constructs domain subclasses and UnsupportedOpError', () => {
    const opErr1 = new UnsupportedOpError('CustomOp');
    expect(opErr1).toBeInstanceOf(Onnx9000Error);
    expect(opErr1.opType).toBe('CustomOp');
    expect(opErr1.message).toBe("Operator 'CustomOp' is not supported yet.");

    const opErr2 = new UnsupportedOpError('CustomOp', 'Explicit message');
    expect(opErr2.message).toBe('Explicit message');

    expect(new ONNXParseError('parse')).toBeInstanceOf(Onnx9000Error);
    expect(new ShapeInferenceError('shape')).toBeInstanceOf(Onnx9000Error);
    expect(new ConversionError('conv')).toBeInstanceOf(Onnx9000Error);
    expect(new BackendError('backend')).toBeInstanceOf(Onnx9000Error);
  });
});
