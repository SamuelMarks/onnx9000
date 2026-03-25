import { Graph } from '@onnx9000/core';
import { extractLlamaMetadata } from './llama';

export function inferArchitecture(graph: Graph): string {
  const name = ((graph.name as any) || '').toLowerCase();
  if (name.includes('mistral')) return 'mistral';
  if (name.includes('mixtral')) return 'mixtral';
  if (name.includes('phi')) return 'phi2';
  if (name.includes('qwen')) return 'qwen2';
  if (name.includes('gemma')) return 'gemma';
  if (name.includes('starcoder')) return 'starcoder';
  if (name.includes('falcon')) return 'falcon';
  if (name.includes('bloom')) return 'bloom';
  if (name.includes('stablelm')) return 'stablelm';
  if (name.includes('command-r')) return 'command-r';
  if (name.includes('bert')) return 'bert';

  let text = Object.keys(graph.tensors).join('') + graph.nodes.map((n) => n.opType).join('');
  if (text.toLowerCase().includes('llama')) return 'llama';

  return 'unknown';
}

export function extractMetadata(graph: Graph, archOverride?: string): Record<string, any> {
  const arch = archOverride || inferArchitecture(graph);
  const validArches = [
    'llama',
    'mistral',
    'mixtral',
    'phi2',
    'qwen2',
    'gemma',
    'starcoder',
    'falcon',
    'bloom',
    'stablelm',
    'command-r',
    'bert',
  ];

  if (archOverride && !validArches.includes(archOverride)) {
    throw new Error(`Unsupported strict architecture mapping: ${archOverride}`);
  }

  if (arch === 'unknown') {
    return {};
  }

  let meta = extractLlamaMetadata(graph);

  if (arch !== 'llama') {
    const remapped: Record<string, any> = {};
    for (const [k, v] of Object.entries(meta)) {
      if (k.startsWith('llama.')) {
        remapped[k.replace('llama.', `${arch}.`)] = v;
      } else {
        remapped[k] = v;
      }
    }
    meta = remapped;
  }

  if (arch === 'mistral') {
    meta['mistral.attention.sliding_window'] = 4096;
  } else if (arch === 'gemma') {
    meta['gemma.attention.layer_norm_rms_epsilon'] = 1e-6;
  } else if (arch === 'mixtral') {
    meta['mixtral.expert_count'] = 8;
    meta['mixtral.expert_used_count'] = 2;
  }

  return meta;
}
