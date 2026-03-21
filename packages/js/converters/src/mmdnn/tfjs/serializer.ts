import { Graph, Tensor } from '@onnx9000/core';

export interface TFJSModelArtifacts {
  modelJson: object;
  weightsBin: Uint8Array;
}

function sanitizeName(name: string): string {
  if (/^[0-9]/.test(name)) return 'v_' + name.replace(/[^a-zA-Z0-9_]/g, '_');
  return name.replace(/[^a-zA-Z0-9_]/g, '_');
}

export function serializeTFJSWeights(graph: Graph): TFJSModelArtifacts {
  const weights: Tensor[] = [];

  for (const initName of graph.initializers) {
    if (graph.tensors[initName]) {
      weights.push(graph.tensors[initName]);
    }
  }

  for (const tName in graph.tensors) {
    const t = graph.tensors[tName];
    if (!t) continue;
    if (t.isInitializer && !weights.includes(t)) {
      weights.push(t);
    }
  }

  let totalBytes = 0;
  for (const w of weights) {
    if (w.data) {
      let byteLength = w.data.byteLength;
      if (byteLength % 4 !== 0) byteLength += 4 - (byteLength % 4);
      totalBytes += byteLength;
    }
  }

  const weightsBin = new Uint8Array(totalBytes);
  let offset = 0;
  const weightsManifestEntries: any[] = [];

  for (const w of weights) {
    if (w.data) {
      const srcBytes = new Uint8Array(w.data.buffer, w.data.byteOffset, w.data.byteLength);
      weightsBin.set(srcBytes, offset);

      const shape = w.shape.map((s) => (s === -1 ? null : s));
      let dtype = 'float32';
      if (w.dtype.includes('int')) dtype = 'int32';
      else if (w.dtype === 'bool') dtype = 'bool';
      else if (w.dtype === 'string') dtype = 'string';

      weightsManifestEntries.push({
        name: sanitizeName(w.name),
        shape: shape,
        dtype: dtype,
      });

      let byteLength = w.data.byteLength;
      if (byteLength % 4 !== 0) byteLength += 4 - (byteLength % 4);
      offset += byteLength;
    }
  }

  const modelJson = {
    format: 'layers-model',
    generatedBy: 'onnx9000.mmdnn',
    convertedBy: 'onnx9000.mmdnn',
    weightsManifest: [
      {
        paths: ['weights.bin'],
        weights: weightsManifestEntries,
      },
    ],
  };

  return {
    modelJson,
    weightsBin,
  };
}
