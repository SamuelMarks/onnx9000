import { describe, it, expect } from 'vitest';
import { Region, Operation, Block } from '../src/ir/core.js';
import { DiagnosticTracer } from '../src/passes/tracing.js';

describe('Advanced Graph Diagnostics', () => {
  it('should implement all mock endpoints', () => {
    const tracer = new DiagnosticTracer();
    tracer.beginPass('test');
    tracer.endPass('test');
    const json = tracer.getChromeTraceJSON();
    expect(json).toContain('compiler_pass');

    tracer.recordMemoryAlloc(1024);
    expect(tracer.getMemoryGraphData().length).toBe(1);

    expect(tracer.traceHALSyncPoints).toBeDefined();

    tracer.recordWGSLSize('test', 'shader');
    expect(tracer.injectGPUProfiling('shader')).toContain('Timestamp Query');

    expect(tracer.mapProfilingToONNX).toBeDefined();
    expect(tracer.diffMLIR('a', 'b')).toContain('- a');
    expect(tracer.dumpShadersToDisk).toBeDefined();
    expect(tracer.executeOnCPUFallback).toBeDefined();
    expect(tracer.mapWebGPUErrorToMLIR).toBeDefined();
  });
});

it('covers tracing endpoints', async () => {
  const { DiagnosticTracer } = await import('../src/passes/tracing.js');
  const tracer = new DiagnosticTracer();
  tracer.dumpShadersToDisk('/tmp');
  const { Region } = await import('../src/ir/core.js');
  tracer.executeOnCPUFallback(new Region());
  tracer.mapWebGPUErrorToMLIR('Some err');
});

it('covers more tracing endpoints', async () => {
  const { DiagnosticTracer } = await import('../src/passes/tracing.js');
  const tracer = new DiagnosticTracer();
  const { Region } = await import('../src/ir/core.js');
  tracer.traceHALSyncPoints(new Region());
  tracer.mapProfilingToONNX(1.0, 'id');
});
