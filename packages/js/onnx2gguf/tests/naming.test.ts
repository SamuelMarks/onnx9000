import { expect, test } from 'vitest';
import { renameTensor } from '../src/naming';

test('renameTensor', () => {
  expect(renameTensor('model.embed_tokens.weight')).toBe('token_embd.weight');
  expect(renameTensor('model.layers.0.input_layernorm.weight')).toBe('blk.0.attn_norm.weight');
  expect(renameTensor('model.layers.10.self_attn.q_proj.weight')).toBe('blk.10.attn_q.weight');
  expect(renameTensor('model.layers.5.self_attn.k_proj.bias')).toBe('blk.5.attn_k.bias');
  expect(renameTensor('model.layers.2.self_attn.v_proj.weight')).toBe('blk.2.attn_v.weight');
  expect(renameTensor('model.layers.1.self_attn.o_proj.weight')).toBe('blk.1.attn_output.weight');
  expect(renameTensor('model.layers.3.self_attn.qkv_proj.weight')).toBe('blk.3.attn_qkv.weight');
  expect(renameTensor('model.layers.4.post_attention_layernorm.weight')).toBe(
    'blk.4.ffn_norm.weight',
  );
  expect(renameTensor('model.layers.6.mlp.gate_proj.weight')).toBe('blk.6.ffn_gate.weight');
  expect(renameTensor('model.layers.7.mlp.down_proj.weight')).toBe('blk.7.ffn_down.weight');
  expect(renameTensor('model.layers.8.mlp.up_proj.weight')).toBe('blk.8.ffn_up.weight');
  expect(renameTensor('model.layers.9.mlp.gate_up_proj.weight')).toBe('blk.9.ffn_gate_up.weight');
  expect(renameTensor('model.layers.1.ffn_gate_inp.weight')).toBe('blk.1.ffn_gate_inp.weight');
  expect(renameTensor('model.norm.weight')).toBe('output_norm.weight');
  expect(renameTensor('lm_head.weight')).toBe('output.weight');

  expect(renameTensor('custom.weight', { '^custom\\.weight$': 'custom_mapped' })).toBe(
    'custom_mapped',
  );

  expect(() => renameTensor('unknown.tensor')).toThrow('Unmatched tensor name: unknown.tensor');
});
