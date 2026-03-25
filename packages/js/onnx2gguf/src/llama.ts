import { Graph, Tensor } from '@onnx9000/core';

export function extractLlamaMetadata(graph: Graph): Record<string, any> {
  const meta: Record<string, any> = {};

  let vocabSize = 32000;
  let hiddenSize = 4096;
  let numHeads = 32;
  let numKvHeads = 32;
  let intermediateSize = 11008;
  let rmsEps = 1e-5;

  for (const [name, t] of Object.entries(graph.tensors)) {
    if (name.endsWith('embed_tokens.weight') && t.shape.length === 2) {
      vocabSize = Number(t.shape[0]);
      hiddenSize = Number(t.shape[1]);
    }
  }

  let headDim = Math.floor(hiddenSize / numHeads);

  for (const [name, t] of Object.entries(graph.tensors)) {
    if (name.endsWith('layers.0.self_attn.q_proj.weight') && t.shape.length === 2) {
      numHeads = Math.floor(Number(t.shape[0]) / headDim);
    } else if (name.endsWith('layers.0.self_attn.k_proj.weight') && t.shape.length === 2) {
      numKvHeads = Math.floor(Number(t.shape[0]) / headDim);
    } else if (name.endsWith('layers.0.mlp.up_proj.weight') && t.shape.length === 2) {
      intermediateSize = Number(t.shape[0]);
    }
  }

  const layers = new Set<number>();
  for (const name of Object.keys(graph.tensors)) {
    const match = name.match(/model\.layers\.(\d+)/);
    if (match) {
      layers.add(parseInt(match[1] || '', 10));
    }
  }
  const blockCount = layers.size > 0 ? Math.max(...Array.from(layers)) + 1 : 32;

  let isSwiglu = false;
  for (const n of graph.nodes) {
    if (n.opType === 'Silu' || n.opType === 'Swish') {
      isSwiglu = true;
    }
    if (n.opType === 'RMSNormalization' && n.attributes && String(n.attributes['epsilon'])) {
      rmsEps = Number(n.attributes['epsilon']);
    }
  }

  meta['llama.context_length'] = 2048;
  meta['llama.embedding_length'] = hiddenSize;
  meta['llama.block_count'] = blockCount;
  meta['llama.feed_forward_length'] = intermediateSize;
  meta['llama.attention.head_count'] = numHeads;
  meta['llama.attention.head_count_kv'] = numKvHeads;
  meta['llama.attention.layer_norm_rms_epsilon'] = rmsEps;
  meta['llama.rope.dimension_count'] = headDim;
  meta['llama.rope.freq_base'] = 10000.0;
  meta['llama.vocab_size'] = vocabSize;
  meta['custom.is_swiglu'] = isSwiglu;

  return meta;
}
