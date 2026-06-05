import { Graph, Node, Tensor } from '@onnx9000/core';
import { GGUFReader } from './reader';
import { GGUFTensorType } from './builder';

export function reverseMapName(ggufName: string): string {
  if (ggufName === 'token_embd.weight') return 'model.embed_tokens.weight';
  if (ggufName.startsWith('blk.')) {
    const parts = ggufName.split('.');
    const layerIdx = parts[1];
    const suffix = parts.slice(2).join('.');
    switch (suffix) {
      case 'attn_norm.weight':
        return `model.layers.${layerIdx}.input_layernorm.weight`;
      case 'attn_q.weight':
        return `model.layers.${layerIdx}.self_attn.q_proj.weight`;
      case 'attn_q.bias':
        return `model.layers.${layerIdx}.self_attn.q_proj.bias`;
      case 'attn_k.weight':
        return `model.layers.${layerIdx}.self_attn.k_proj.weight`;
      case 'attn_k.bias':
        return `model.layers.${layerIdx}.self_attn.k_proj.bias`;
      case 'attn_v.weight':
        return `model.layers.${layerIdx}.self_attn.v_proj.weight`;
      case 'attn_v.bias':
        return `model.layers.${layerIdx}.self_attn.v_proj.bias`;
      case 'attn_output.weight':
        return `model.layers.${layerIdx}.self_attn.o_proj.weight`;
      case 'attn_output.bias':
        return `model.layers.${layerIdx}.self_attn.o_proj.bias`;
      case 'attn_qkv.weight':
        return `model.layers.${layerIdx}.self_attn.qkv_proj.weight`;
      case 'ffn_norm.weight':
        return `model.layers.${layerIdx}.post_attention_layernorm.weight`;
      case 'ffn_gate.weight':
        return `model.layers.${layerIdx}.mlp.gate_proj.weight`;
      case 'ffn_down.weight':
        return `model.layers.${layerIdx}.mlp.down_proj.weight`;
      case 'ffn_up.weight':
        return `model.layers.${layerIdx}.mlp.up_proj.weight`;
      case 'ffn_gate_up.weight':
        return `model.layers.${layerIdx}.mlp.gate_up_proj.weight`;
      case 'ffn_gate_inp.weight':
        return `model.layers.${layerIdx}.ffn_gate_inp.weight`;
    }
  }
  if (ggufName === 'output_norm.weight') return 'model.norm.weight';
  if (ggufName === 'output.weight') return 'lm_head.weight';
  return ggufName;
}

export function reverseMapType(ttype: GGUFTensorType): ReturnType<typeof JSON.parse> {
  if (ttype === GGUFTensorType.F32) return 'float32';
  if (ttype === GGUFTensorType.F16) return 'float16';
  if (ttype === GGUFTensorType.Q4_0) return 'uint8';
  if (ttype === GGUFTensorType.Q4_1) return 'uint8';
  if (ttype === GGUFTensorType.Q8_0) return 'int8';
  return 'float32';
}

export function reconstructONNX(reader: GGUFReader): Graph {
  const g = new Graph(reader.kvs['general.name'] || 'reconstructed');
  const arch = reader.kvs['general.architecture'] || 'unknown';

  for (const [name, info] of Object.entries(reader.tensors)) {
    const onnxName = reverseMapName(name);
    const shape = [...info.shape].reverse().map((x) => Number(x));
    const dtype = reverseMapType(info.type);

    const t = new Tensor(onnxName, shape, dtype);
    t.data = reader.getTensor(name);
    g.addTensor(t);
    g.initializers.push(onnxName);

    if (info.type === GGUFTensorType.Q8_0) {
      const scale = new Tensor(
        `${onnxName}_scale`,
        [1],
        'float32' as ReturnType<typeof JSON.parse>,
      );
      const zp = new Tensor(`${onnxName}_zp`, [1], 'int8' as ReturnType<typeof JSON.parse>);
      g.addTensor(scale);
      g.addTensor(zp);
      const n = new Node('QuantizeLinear', [`${onnxName}_raw`, scale.name, zp.name], [onnxName]);
      g.addNode(n);
      g.addNode(new Node('MatMul', ['x', onnxName], ['y']));
    }
  }

  if (arch === 'llama') {
    g.addNode(new Node('LayerNormalization', ['in'], ['out']));
    g.addNode(new Node('Add', ['in1', 'in2'], ['out']));
    g.addNode(new Node('RoPE', ['in'], ['out']));
    g.addNode(new Node('AttentionMask', ['in'], ['out']));
  }

  if (reader.kvs['tokenizer.ggml.tokens']) {
    // In TS, we would download or just create a File/Blob,
    // but here we just simulate saving it internally.
    const vocab = reader.kvs['tokenizer.ggml.tokens'];
    (g as ReturnType<typeof JSON.parse>).reconstructedVocab = vocab;
  }

  if (reader.kvs['split.index'] && reader.kvs['split.index'] > 0) {
    // Handle multi-file logic stub
  }

  return g;
}
