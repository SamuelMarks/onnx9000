import { describe, expect, it } from 'vitest';
import { TritonServer } from '../src/index.js';

describe('TritonServer', () => {
  it('should generate config.pbtxt and process requests', () => {
    const srv = new TritonServer();
    expect(srv.process('resnet50')).toBe('Triton Server processed resnet50');
    const cfg = srv.generateConfigPbtxt(
      'bert',
      16,
      [{ name: 'ids', dataType: 'TYPE_INT64', dims: [128] }],
      [{ name: 'logits', dataType: 'TYPE_FP32', dims: [2] }],
    );
    expect(cfg).toContain('name: "bert"');
    expect(cfg).toContain('max_batch_size: 16');
  });
});
