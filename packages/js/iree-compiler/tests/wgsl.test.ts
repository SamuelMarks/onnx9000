import { describe, it, expect } from 'vitest';
import { Type, Value, Region, Operation, Block } from '../src/ir/core.js';
import { WGSLEmitter } from '../src/passes/lower_wgsl.js';

describe('WGSL Lowering', () => {
  it('should emit minified wgsl', () => {
    const region = new Region();
    const block = new Block(region);
    region.pushBlock(block);

    block.pushOperation(new Operation('web.linalg.matmul', [], [], {}));
    block.pushOperation(new Operation('web.mhlo.maximum', [], [], {}));

    const emitter = new WGSLEmitter();
    const wgsl = emitter.emit(region, { fp16: true, workgroupSize: [16, 16, 1] });

    // 118, 119
    expect(wgsl).toContain('enable f16;');
    expect(wgsl).toContain('@compute @workgroup_size(16,16,1)');
    // 109, 111, 113, 117
    expect(wgsl).toContain('fn main(@builtin(global_invocation_id) global_id:vec3<u32>)');
    expect(wgsl).toContain('var<workgroup> tile_a:array<f32,256>;');
    expect(wgsl).toContain('let flat_idx = global_id.y * 64u + global_id.x;');
  });
});
