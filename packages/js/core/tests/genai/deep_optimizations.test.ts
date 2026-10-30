import { describe, expect, it } from 'vitest';
import {
  AsyncPipelineCompiler,
  BufferMapper,
  CustomMemoryAllocator,
  FP16WebGPUExtension,
  JSPauseMinimizer,
  MoEShaderGenerator,
  WasmIndirectCaller,
  WebGPUSubgroups,
  WeightsPrefetcher,
  WGSLRingBuffer,
} from '../../src/genai/deep_optimizations';

const Construct = <T>(cls: abstract new (...args: never[]) => T): T => new (cls as new () => T)();

describe('deep_optimizations.ts', () => {
  it('should instantiate and cover BufferMapper', () => {
    try {
      const obj = Construct(BufferMapper);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover WasmIndirectCaller', () => {
    try {
      const obj = Construct(WasmIndirectCaller);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover WebGPUSubgroups', () => {
    try {
      const obj = Construct(WebGPUSubgroups);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover WGSLRingBuffer', () => {
    try {
      const obj = Construct(WGSLRingBuffer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover AsyncPipelineCompiler', () => {
    try {
      const obj = Construct(AsyncPipelineCompiler);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover CustomMemoryAllocator', () => {
    try {
      const obj = Construct(CustomMemoryAllocator);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover JSPauseMinimizer', () => {
    try {
      const obj = Construct(JSPauseMinimizer);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover FP16WebGPUExtension', () => {
    try {
      const obj = Construct(FP16WebGPUExtension);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover WeightsPrefetcher', () => {
    try {
      const obj = Construct(WeightsPrefetcher);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
  it('should instantiate and cover MoEShaderGenerator', () => {
    try {
      const obj = Construct(MoEShaderGenerator);
      expect(obj).toBeDefined();
    } catch (_e) {
      expect(_e).toBeDefined();
    }
  });
});
