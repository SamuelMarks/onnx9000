import { renameTensor } from './naming';
import { extractTokenizerMetadata } from './tokenizer';
import { extractMetadata, inferArchitecture } from './arch';
import { Graph, Tensor } from '@onnx9000/core';
import { GGUFWriter, GGUFValueType, GGUFTensorType } from './builder';

function getGGUFType(dtype: string): GGUFTensorType {
  if (dtype === 'float32') return GGUFTensorType.F32;
  if (dtype === 'float16') return GGUFTensorType.F16;
  return GGUFTensorType.F32;
}

function sanitizeDocString(doc: string): string {
  return doc ? doc.trim() : '';
}

export function compileGGUF(
  graph: Graph,
  kvOverrides: Record<string, any> = {},
  archOverride?: string,
): ArrayBuffer {
  const writer = new GGUFWriter();

  const arch = archOverride || inferArchitecture(graph);

  // Phase 2 Defaults
  const generalKvs: Record<string, any> = {
    'general.architecture': arch,
    'general.name': graph.name || 'model',
    'general.author': (graph as any).producerName || 'onnx9000',
    'general.version': Number((graph as any).modelVersion || 1),
    'general.quantization_version': 2,
    'general.alignment': 32,
    'general.file_type': 'mostly_f32',
  };

  const docStr = (graph as any).docString;
  if (docStr) {
    generalKvs['general.description'] = sanitizeDocString(docStr);
  }

  for (const [k, v] of Object.entries(kvOverrides)) {
    if (k.startsWith('general.')) {
      generalKvs[k] = v;
    }
  }

  for (const [k, v] of Object.entries(generalKvs)) {
    if (typeof v === 'boolean') writer.addBool(k, v);
    else if (typeof v === 'string') writer.addString(k, v);
    else if (typeof v === 'number' && !Number.isInteger(v)) writer.addFloat32(k, v);
    else if (typeof v === 'number') writer.addUint32(k, v);
  }

  // Phase 3 & 4
  const archMeta = extractMetadata(graph, archOverride);
  for (const [k, v] of Object.entries(archMeta)) {
    if (typeof v === 'boolean') writer.addBool(k, v);
    else if (typeof v === 'string') writer.addString(k, v);
    else if (typeof v === 'number' && !Number.isInteger(v)) writer.addFloat32(k, v);
    else if (typeof v === 'number') writer.addUint32(k, v);
  }

  // Phase 5: Tokenizer
  const tokMeta = extractTokenizerMetadata(
    kvOverrides['tokenizer.json'] || null,
    archMeta['llama.vocab_size'] || 0,
  );
  for (const [k, v] of Object.entries(tokMeta)) {
    if (!kvOverrides[k]) {
      if (typeof v === 'boolean') writer.addBool(k, v);
      else if (typeof v === 'string') writer.addString(k, v);
      else if (Array.isArray(v))
        writer.addArray(
          k,
          v,
          typeof v[0] === 'number'
            ? GGUFValueType.FLOAT32
            : typeof v[0] === 'string'
              ? GGUFValueType.STRING
              : /* v8 ignore start */
                GGUFValueType.INT32,
          /* v8 ignore stop */
        );
      else if (typeof v === 'number') writer.addUint32(k, v);
    }
  }

  for (const [k, v] of Object.entries(kvOverrides)) {
    if (!k.startsWith('general.') && k !== 'tokenizer.json') {
      if (typeof v === 'boolean') writer.addBool(k, v);
      else if (typeof v === 'string') writer.addString(k, v);
      else if (Array.isArray(v))
        writer.addArray(
          k,
          v,
          typeof v[0] === 'number'
            ? GGUFValueType.FLOAT32
            : typeof v[0] === 'string'
              ? GGUFValueType.STRING
              : GGUFValueType.INT32,
        );
      else if (typeof v === 'number' && !Number.isInteger(v)) writer.addFloat32(k, v);
      else if (typeof v === 'number') writer.addUint32(k, v);
    }
  }

  // Phase 6 & 7: Tensors

  let currentOffset = 0n;
  const tensorDefs: { name: string; shape: bigint[]; type: GGUFTensorType; tensor: Tensor }[] = [];

  for (const initName of graph.initializers) {
    const t = graph.tensors[initName];
    if (t) {
      const ggufName = initName.replace(/model\.layers\.(\d+)/g, 'blk.$1');
      const shape = t.shape.map((s) => (typeof s === 'number' ? BigInt(s) : 1n)).reverse();
      const type = getGGUFType(t.dtype);
      writer.addTensorInfo(ggufName, shape, type, currentOffset);
      tensorDefs.push({ name: ggufName, shape, type, tensor: t });

      let items = 1n;
      for (const d of shape) items *= d;
      let size = 0n;
      if (type === GGUFTensorType.F32) size = items * 4n;
      else if (type === GGUFTensorType.F16) size = items * 2n;

      let alignment = 32n;
      for (const kv of writer.kvs) {
        if (kv.key === 'general.alignment' && kv.type === GGUFValueType.UINT32) {
          alignment = BigInt(kv.val);
        }
      }

      const padding = (alignment - (size % alignment)) % alignment;
      currentOffset += size + padding;
    }
  }

  const headerSize = writer.getHeaderSize();
  const totalSize = BigInt(headerSize) + currentOffset;

  const buffer = new ArrayBuffer(Number(totalSize));
  const written = writer.writeHeader(buffer);
  let cursor = written;
  const u8View = new Uint8Array(buffer);

  for (const t of tensorDefs) {
    if (t.tensor.data) {
      u8View.set(
        new Uint8Array(t.tensor.data.buffer, t.tensor.data.byteOffset, t.tensor.data.byteLength),
        cursor,
      );
      let items = 1n;
      for (const d of t.shape) items *= d;
      let size = 0n;
      if (t.type === GGUFTensorType.F32) size = items * 4n;
      else if (t.type === GGUFTensorType.F16) size = items * 2n;

      let alignment = 32n;
      for (const kv of writer.kvs) {
        if (kv.key === 'general.alignment' && kv.type === GGUFValueType.UINT32) {
          alignment = BigInt(kv.val);
        }
      }

      const padding = (alignment - (size % alignment)) % alignment;
      cursor += Number(size + padding);
    }
  }

  return buffer;
}
