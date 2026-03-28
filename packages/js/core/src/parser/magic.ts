export type ModelFormat =
  | 'onnx'
  | 'pb'
  | 'tflite'
  | 'h5'
  | 'pt'
  | 'mlmodel'
  | 'pkl'
  | 'pdmodel'
  | 'safetensors'
  | 'unknown';

export async function detectFormat(file: File | Blob): Promise<ModelFormat> {
  if (file.size < 4) return 'unknown';

  const slice = file.slice(0, 8);
  const buffer = await slice.arrayBuffer();
  const bytes = new Uint8Array(buffer);

  // ONNX/Protobuf usually don't have a rigid magic number, but we can look for specific heuristics.
  // Actually, some formats do.

  // TFLite: 'TFL3' at offset 4
  if (
    bytes.length >= 8 &&
    bytes[4] === 0x54 &&
    bytes[5] === 0x46 &&
    bytes[6] === 0x4c &&
    bytes[7] === 0x33
  ) {
    return 'tflite';
  }

  // HDF5 (Keras): \x89HDF\r\n\x1a\n
  if (bytes[0] === 0x89 && bytes[1] === 0x48 && bytes[2] === 0x44 && bytes[3] === 0x46) {
    return 'h5';
  }

  // PyTorch (Zip format typically starts with PK\x03\x04)
  if (bytes[0] === 0x50 && bytes[1] === 0x4b && bytes[2] === 0x03 && bytes[3] === 0x04) {
    // Actually, MLModel is also a zip sometimes, or has its own structure.
    // For now we assume a zip might be PyTorch (often .pt or .pth)
    if ('name' in file) {
      const f = file;
      if (f.name.endsWith('.pt') || f.name.endsWith('.pth')) return 'pt';
    }
  }

  // Safetensors: Starts with 8-byte little-endian header length
  // Then JSON. We can heuristically check if it ends in safetensors
  if ('name' in file) {
    const f = file;
    if (f.name.endsWith('.safetensors')) return 'safetensors';
  }

  // Default fallback for strictly ONNX
  if ('name' in file) {
    const f = file;
    if (f.name.endsWith('.onnx')) return 'onnx';
  }

  return 'unknown';
}
