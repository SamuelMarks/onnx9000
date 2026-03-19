import { test, expect } from 'vitest';
import * as mocks from '../src/safetensors_mocks';

test('safetensors_mocks all functions can be called', () => {
  mocks.handleMultiGigabyteWasm32Models();
  mocks.emulate64BitMemoryAddressing();
  mocks.integrateWebGPUChunkedPipeline();
  mocks.tensorParallelismLoadSlice({});
  mocks.appendBinaryBuffersWritev();
  mocks.streamArraysSequentially();
  mocks.validateRustByteParity();
  for (const v of mocks.yieldStreamSerialization()) {
    expect(v).toBeInstanceOf(Uint8Array);
  }
  mocks.validateHubEtag();
  mocks.benchmark1GbLayerStream();
  mocks.profileGarbageCollectionV8();
  mocks.monitorHttpKeepAlive();
  mocks.profileHttpRangeRequests();
});
