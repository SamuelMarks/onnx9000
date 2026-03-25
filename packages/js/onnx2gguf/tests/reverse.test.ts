import { expect, test } from 'vitest';
import { reverseMapName, reverseMapType, reconstructONNX } from '../src/reverse';
import { GGUFWriter, GGUFValueType, GGUFTensorType } from '../src/builder';
import { GGUFReader } from '../src/reader';

test('reverse mapping functions', () => {
  expect(reverseMapName('token_embd.weight')).toBe('model.embed_tokens.weight');
  expect(reverseMapName('blk.0.attn_norm.weight')).toBe('model.layers.0.input_layernorm.weight');
  expect(reverseMapName('blk.1.attn_q.weight')).toBe('model.layers.1.self_attn.q_proj.weight');
  expect(reverseMapName('blk.2.attn_q.bias')).toBe('model.layers.2.self_attn.q_proj.bias');
  expect(reverseMapName('blk.3.attn_k.weight')).toBe('model.layers.3.self_attn.k_proj.weight');
  expect(reverseMapName('blk.4.attn_k.bias')).toBe('model.layers.4.self_attn.k_proj.bias');
  expect(reverseMapName('blk.5.attn_v.weight')).toBe('model.layers.5.self_attn.v_proj.weight');
  expect(reverseMapName('blk.6.attn_v.bias')).toBe('model.layers.6.self_attn.v_proj.bias');
  expect(reverseMapName('blk.7.attn_output.weight')).toBe('model.layers.7.self_attn.o_proj.weight');
  expect(reverseMapName('blk.8.attn_output.bias')).toBe('model.layers.8.self_attn.o_proj.bias');
  expect(reverseMapName('blk.9.attn_qkv.weight')).toBe('model.layers.9.self_attn.qkv_proj.weight');
  expect(reverseMapName('blk.10.ffn_norm.weight')).toBe(
    'model.layers.10.post_attention_layernorm.weight',
  );
  expect(reverseMapName('blk.11.ffn_gate.weight')).toBe('model.layers.11.mlp.gate_proj.weight');
  expect(reverseMapName('blk.12.ffn_down.weight')).toBe('model.layers.12.mlp.down_proj.weight');
  expect(reverseMapName('blk.13.ffn_up.weight')).toBe('model.layers.13.mlp.up_proj.weight');
  expect(reverseMapName('blk.14.ffn_gate_up.weight')).toBe(
    'model.layers.14.mlp.gate_up_proj.weight',
  );
  expect(reverseMapName('blk.15.ffn_gate_inp.weight')).toBe('model.layers.15.ffn_gate_inp.weight');
  expect(reverseMapName('output_norm.weight')).toBe('model.norm.weight');
  expect(reverseMapName('output.weight')).toBe('lm_head.weight');
  expect(reverseMapName('unknown')).toBe('unknown');

  expect(reverseMapType(GGUFTensorType.F32)).toBe('float32');
  expect(reverseMapType(GGUFTensorType.F16)).toBe('float16');
  expect(reverseMapType(GGUFTensorType.Q4_0)).toBe('uint8');
  expect(reverseMapType(GGUFTensorType.Q4_1)).toBe('uint8');
  expect(reverseMapType(GGUFTensorType.Q8_0)).toBe('int8');
  expect(reverseMapType(-1 as GGUFTensorType)).toBe('float32');
});

test('reconstructONNX', () => {
  const writer = new GGUFWriter();
  writer.addString('general.name', 'test');
  writer.addString('general.architecture', 'llama');
  writer.addUint32('split.index', 1);
  writer.addArray('tokenizer.ggml.tokens', ['a', 'b'], GGUFValueType.STRING);
  writer.addTensorInfo('token_embd.weight', [2n, 2n], GGUFTensorType.F32, 0n);
  writer.addTensorInfo('blk.0.attn_q.weight', [32n], GGUFTensorType.Q8_0, 32n);

  const buf = new ArrayBuffer(500);
  writer.writeHeader(buf);
  const view = new Uint8Array(buf);
  const off = writer.writeHeader(buf);
  view.set(new Uint8Array(16), off);
  view.set(new Uint8Array(34), off + 32);

  const reader = new GGUFReader(buf);
  const g = reconstructONNX(reader);

  expect(g.name).toBe('test');
  console.log(Object.keys(g.tensors));
  expect('model.embed_tokens.weight' in g.tensors).toBe(true);
  expect('model.layers.0.self_attn.q_proj.weight' in g.tensors).toBe(true);

  const opTypes = g.nodes.map((n) => n.opType);
  expect(opTypes).toContain('LayerNormalization');
  expect(opTypes).toContain('Add');
  expect(opTypes).toContain('RoPE');
  expect(opTypes).toContain('AttentionMask');
  expect(opTypes).toContain('QuantizeLinear');
  expect(opTypes).toContain('MatMul');

  expect((g as any).reconstructedVocab).toEqual(['a', 'b']);
});
