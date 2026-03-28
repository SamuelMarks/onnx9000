import { Graph } from '@onnx9000/core';
import { Tensor, SparseTensor } from '@onnx9000/core';
import { unpackData } from '@onnx9000/core';

export abstract class Modifier {
  constructor(public options: any = {}) {}
  abstract apply(graph: Graph): void;
}

export class MagnitudePruningModifier extends Modifier {
  apply(graph: Graph): void {
    const sparsity = this.options.final_sparsity || 0;
    const params = this.options.params || [];
    const leaveUnmasked = this.options.leave_unmasked || [];

    for (const pattern of params) {
      const regex = new RegExp(pattern.replace('re:', ''));
      for (const name in graph.tensors) {
        if (leaveUnmasked.includes(name)) continue;
        const tensor = graph.tensors[name]!;
        if (tensor.isInitializer && regex.test(name)) {
          this._pruneBySparsity(tensor, sparsity);
        }
      }
    }
  }

  private _pruneBySparsity(tensor: Tensor, sparsity: number): void {
    if (!tensor.data) return;
    const values = unpackData(tensor) as number[];
    if (values.length === 0) return;

    const absValues = values.map((v) => Math.abs(v)).sort((a, b) => a - b);
    const idx = Math.floor(absValues.length * sparsity) - 1;
    const threshold = idx >= 0 ? absValues[idx]! : -1;

    // We mutate the data in place or create new typed array
    // @ts-ignore
    for (let i = 0; i < values.length; i++) {
      if (Math.abs(values[i]!) <= threshold) {
        // @ts-ignore
        tensor.data[i] = 0;
      }
    }
  }
}

export class ConstantPruningModifier extends Modifier {
  apply(graph: Graph): void {
    const params = this.options.params || [];
    for (const pattern of params) {
      const regex = new RegExp(pattern.replace('re:', ''));
      for (const name in graph.tensors) {
        const tensor = graph.tensors[name]!;
        if (tensor.isInitializer && regex.test(name)) {
          this._pruneByThreshold(tensor, 0);
        }
      }
    }
  }

  private _pruneByThreshold(tensor: Tensor, threshold: number): void {
    if (!tensor.data) return;
    const values = unpackData(tensor) as number[];
    // @ts-ignore
    for (let i = 0; i < values.length; i++) {
      if (Math.abs(values[i]!) <= threshold) {
        // @ts-ignore
        tensor.data[i] = 0;
      }
    }
  }
}

export function parseRecipe(yamlText: string): Modifier[] {
  const modifiers: Modifier[] = [];
  const lines = yamlText.split('\n');
  let currentMod: any = null;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;

    if (trimmed.startsWith('- !')) {
      const type = trimmed.substring(3).split(' ')[0]!;
      currentMod = { type, options: {} };
      if (type === 'MagnitudePruningModifier')
        modifiers.push(new MagnitudePruningModifier(currentMod.options));
      else if (type === 'ConstantPruningModifier')
        modifiers.push(new ConstantPruningModifier(currentMod.options));
      else
        modifiers.push(
          new (class extends Modifier {
            apply() {}
          })(currentMod.options),
        );
    } else if (currentMod && trimmed.includes(':')) {
      const parts = trimmed.split(':');
      const key = parts[0]!.trim();
      let val: any = parts.slice(1).join(':').trim();

      if (val.startsWith('[') && val.endsWith(']')) {
        val = val
          .substring(1, val.length - 1)
          .split(',')
          .map((s: string) => s.trim().replace(/['"]/g, ''));
      } else if (!isNaN(val)) {
        val = parseFloat(val);
      } else {
        val = val.replace(/['"]/g, '');
      }
      currentMod.options[key] = val;
    }
  }
  return modifiers;
}

export function applyRecipe(graph: Graph, recipeYaml: string): void {
  const modifiers = parseRecipe(recipeYaml);
  for (const mod of modifiers) {
    mod.apply(graph);
  }
  graph.metadataProps['onnx9000_sparse_recipe'] = recipeYaml;
}
