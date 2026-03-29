/* eslint-disable */
// @ts-nocheck
import { WeightGroup, DataType, WeightManifestEntry } from './tfjs-parser.js';

export interface LoadedWeight {
  name: string;
  shape: number[];
  dtype: DataType;
  data: ArrayBuffer;
}

export type FetchFunction = (path: string) => Promise<ArrayBuffer>;

/**
 * Downloads external binary weight shards for TF.js.
 */
export async function downloadWeightShards(
  manifest: WeightGroup[],
  baseUrl: string,
  fetcher?: FetchFunction,
): Promise<LoadedWeight[]> {
  const loadedWeights: LoadedWeight[] = [];
  const defaultFetcher: FetchFunction = async (path: string) => {
    const url = new URL(path, baseUrl).href;
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`Failed to fetch weight shard: ${url}`);
    }
    return res.arrayBuffer();
  };

  const doFetch = fetcher || defaultFetcher;

  for (const group of manifest) {
    // Download all shards for this group
    const shardBuffers = await Promise.all(group.paths.map((path) => doFetch(path)));

    // Combine chunked .bin shards into a contiguous ArrayBuffer
    let totalLength = 0;
    for (const buf of shardBuffers) {
      totalLength += buf.byteLength;
    }

    const combinedBuffer = new Uint8Array(totalLength);
    let offset = 0;
    for (const buf of shardBuffers) {
      combinedBuffer.set(new Uint8Array(buf), offset);
      offset += buf.byteLength;
    }

    // Map TF.js weight manifests to specific layer variables
    let byteOffset = 0;
    for (const weightMeta of group.weights) {
      const byteLength = calculateByteLength(weightMeta);
      const weightData = combinedBuffer.buffer.slice(
        combinedBuffer.byteOffset + byteOffset,
        combinedBuffer.byteOffset + byteOffset + byteLength,
      );
      loadedWeights.push({
        name: weightMeta.name,
        shape: weightMeta.shape,
        dtype: weightMeta.dtype,
        data: weightData,
      });
      byteOffset += byteLength;
    }
  }

  return loadedWeights;
}

function calculateByteLength(weight: WeightManifestEntry): number {
  const numElements = weight.shape.reduce((a, b) => a * b, 1);
  switch (weight.dtype) {
    case 'float32':
    case 'int32':
      return numElements * 4;
    case 'complex64':
      return numElements * 8;
    case 'float16':
      return numElements * 2;
    case 'uint8':
    case 'bool':
      return numElements;
    case 'string':
      // String tensors in TFJS have a specific format in the buffer.
      // A simple size calculation doesn't work out of the box without parsing.
      // For now, let's assume we won't fully extract strings in this naive pass
      // or we throw an error for unsupported format if it happens.
      throw new Error('String dtype byte length calculation is not trivially supported yet.');
    default:
      throw new Error(`Unsupported dtype: ${weight.dtype}`);
  }
}
