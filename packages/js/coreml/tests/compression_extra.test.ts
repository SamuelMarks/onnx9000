import { describe, it, expect } from 'vitest';
import { Builder, MILDataType, applyCompression, TensorType } from '../src/index.js';

describe('Compression Extras', () => {
  it('Handles mixed precision dictionary and gathers stats', () => {
    const builder = new Builder();
    builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');

    const x = builder.createVar('w1', builder.tensor(MILDataType.FLOAT32, [100]));
    const y = builder.createVar('w2', builder.tensor(MILDataType.FLOAT32, [100]));
    const z = builder.createVar('w3', builder.tensor(MILDataType.FLOAT32, [100]));

    builder.addOp('const', {}, [x], { value: new Float32Array(100) });
    builder.addOp('const', {}, [y], { value: new Float32Array(100) });
    builder.addOp('const', {}, [z], { value: new Float32Array(100) });

    // Test native map QLinear
    builder.addOp('quantize_linear', { x }, [x]);

    // Test KV Cache
    builder.addOp('read_state', { x }, [x]);

    applyCompression(block, {
      mode: 'mixed',
      mixedPrecisionConfig: {
        w1: 'w8a16',
        w2: 'w4a16',
        // w3 stays fp32 (no quant) since mixed is requested but no key found (falls back to mode 'mixed' which has no explicit implementation outside w8/w4/sparse)
      },
      multiBitrate: true,
      gatherStatistics: true,
      kvCacheQuantization: true,
      reportReduction: false,
    });

    const ops = block.operations;
    // w1 const
    expect(ops[0]!.opType).toBe('constexpr_affine_dequantize');
    expect(ops[0]!.attributes['quant_type']).toBe('int8');
    expect(ops[0]!.attributes['multi_bitrate_enabled']).toBe(true);
    expect(ops[0]!.attributes['ane_hint_dynamic_quant_stats_gathered']).toBe(true);

    // w2 const
    expect(ops[1]!.opType).toBe('constexpr_affine_dequantize');
    expect(ops[1]!.attributes['quant_type']).toBe('int4');

    // w3 const
    expect(ops[2]!.opType).toBe('const'); // untouched

    // quantize_linear
    expect(ops[3]!.attributes['ane_hint_mapped_qlinear']).toBe(true);

    // read_state
    expect(ops[4]!.attributes['kv_cache_quantized']).toBe('int4');
  });

  it('safely skips compression for non-tensor outputs', () => {
    const builder = new Builder();
    builder.createFunction('test', [], []);
    const block = builder.createBlock('block0');
    const x = builder.createVar('w1', builder.scalar(MILDataType.FLOAT32));
    builder.addOp('const', {}, [x], { value: 1.0 });

    applyCompression(block, { mode: 'w8a16' });
    expect(block.operations[0]!.opType).toBe('const');
  });
});
