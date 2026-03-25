const DEFAULT_MAPPING: [RegExp, string][] = [
  [/^model\.embed_tokens\.weight$/, 'token_embd.weight'],
  [/^model\.layers\.(\d+)\.input_layernorm\.weight$/, 'blk.$1.attn_norm.weight'],
  [/^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$/, 'blk.$1.attn_q.weight'],
  [/^model\.layers\.(\d+)\.self_attn\.q_proj\.bias$/, 'blk.$1.attn_q.bias'],
  [/^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$/, 'blk.$1.attn_k.weight'],
  [/^model\.layers\.(\d+)\.self_attn\.k_proj\.bias$/, 'blk.$1.attn_k.bias'],
  [/^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$/, 'blk.$1.attn_v.weight'],
  [/^model\.layers\.(\d+)\.self_attn\.v_proj\.bias$/, 'blk.$1.attn_v.bias'],
  [/^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$/, 'blk.$1.attn_output.weight'],
  [/^model\.layers\.(\d+)\.self_attn\.o_proj\.bias$/, 'blk.$1.attn_output.bias'],
  [/^model\.layers\.(\d+)\.self_attn\.qkv_proj\.weight$/, 'blk.$1.attn_qkv.weight'],
  [/^model\.layers\.(\d+)\.post_attention_layernorm\.weight$/, 'blk.$1.ffn_norm.weight'],
  [/^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$/, 'blk.$1.ffn_gate.weight'],
  [/^model\.layers\.(\d+)\.mlp\.down_proj\.weight$/, 'blk.$1.ffn_down.weight'],
  [/^model\.layers\.(\d+)\.mlp\.up_proj\.weight$/, 'blk.$1.ffn_up.weight'],
  [/^model\.layers\.(\d+)\.mlp\.gate_up_proj\.weight$/, 'blk.$1.ffn_gate_up.weight'],
  [/^model\.layers\.(\d+)\.ffn_gate_inp\.weight$/, 'blk.$1.ffn_gate_inp.weight'],
  [/^model\.norm\.weight$/, 'output_norm.weight'],
  [/^lm_head\.weight$/, 'output.weight'],
];

export function renameTensor(name: string, overrides: Record<string, string> = {}): string {
  for (const [patternStr, repl] of Object.entries(overrides)) {
    const regex = new RegExp(patternStr);
    if (regex.test(name)) {
      return name.replace(regex, repl);
    }
  }

  for (const [regex, repl] of DEFAULT_MAPPING) {
    if (regex.test(name)) {
      return name.replace(regex, repl);
    }
  }

  throw new Error(`Unmatched tensor name: ${name}`);
}
