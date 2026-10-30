/**
 * @fileoverview modifier.ts
 * Provides modifier functionality for the modifier package.
 */
import { type Graph, type Tensor, unpackData } from '@onnx9000/core';

/** Type alias `WritableNumericArray`. */
type WritableNumericArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | number[];

/**
 * Executes `zeroTensorElement`.
 * @param data Parameter `data`.
 * @param index Parameter `index`.
 */
function zeroTensorElement(data: ArrayBufferView | number[], index: number): void {
  (data as WritableNumericArray)[index] = 0;
}

/** Represents the `Modifier` class. */
export abstract class Modifier {
  /**
   * Initializes a new `Modifier` instance.
   * @param options Parameter `options`.
   */
  constructor(public options: ReturnType<typeof JSON.parse> = {}) {}
  /**
   * Executes `apply` on `Modifier`.
   * @param graph Parameter `graph`.
   */
  abstract apply(graph: Graph): void;
}

/** Represents the `MagnitudePruningModifier` class. */
export class MagnitudePruningModifier extends Modifier {
  /**
   * Executes `apply` on `MagnitudePruningModifier`.
   * @param graph Parameter `graph`.
   */
  apply(graph: Graph): void {
    const sparsity = this.options.final_sparsity || 0;
    const params = this.options.params || [];
    const leaveUnmasked = this.options.leave_unmasked || [];

    for (const pattern of params) {
      const regex = new RegExp(pattern.replace('re:', ''));
      for (const name in graph.tensors) {
        if (leaveUnmasked.includes(name)) continue;
        const tensor = graph.tensors[name] as Tensor;
        if (tensor.isInitializer && regex.test(name)) {
          this._pruneBySparsity(tensor, sparsity);
        }
      }
    }
  }

  /**
   * Executes `_pruneBySparsity` on `MagnitudePruningModifier`.
   * @param tensor Parameter `tensor`.
   * @param sparsity Parameter `sparsity`.
   */
  private _pruneBySparsity(tensor: Tensor, sparsity: number): void {
    if (!tensor.data) return;
    const values = unpackData(tensor) as number[];
    if (values.length === 0) return;

    const absValues = values.map((v) => Math.abs(v)).sort((a, b) => a - b);
    const idx = Math.floor(absValues.length * sparsity) - 1;
    const threshold = idx >= 0 ? (absValues[idx] as number) : -1;

    // We mutate the data in place or create new typed array
    for (let i = 0; i < values.length; i++) {
      if (Math.abs(values[i] as number) <= threshold) {
        zeroTensorElement(tensor.data, i);
      }
    }
  }
}

/** Represents the `ConstantPruningModifier` class. */
export class ConstantPruningModifier extends Modifier {
  /**
   * Executes `apply` on `ConstantPruningModifier`.
   * @param graph Parameter `graph`.
   */
  apply(graph: Graph): void {
    const params = this.options.params || [];
    for (const pattern of params) {
      const regex = new RegExp(pattern.replace('re:', ''));
      for (const name in graph.tensors) {
        const tensor = graph.tensors[name] as Tensor;
        if (tensor.isInitializer && regex.test(name)) {
          this._pruneByThreshold(tensor, 0);
        }
      }
    }
  }

  /**
   * Executes `_pruneByThreshold` on `ConstantPruningModifier`.
   * @param tensor Parameter `tensor`.
   * @param threshold Parameter `threshold`.
   */
  private _pruneByThreshold(tensor: Tensor, threshold: number): void {
    if (!tensor.data) return;
    const values = unpackData(tensor) as number[];
    for (let i = 0; i < values.length; i++) {
      if (Math.abs(values[i] as number) <= threshold) {
        zeroTensorElement(tensor.data, i);
      }
    }
  }
}

/**
 * Executes `parseRecipe`.
 * @param yamlText Parameter `yamlText`.
 * @returns Result of `parseRecipe`.
 */
export function parseRecipe(yamlText: string): Modifier[] {
  const modifiers: Modifier[] = [];
  const lines = yamlText.split('\n');
  let currentMod: ReturnType<typeof JSON.parse> = null;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;

    if (trimmed.startsWith('- !')) {
      const type = trimmed.substring(3).split(' ')[0];
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
      const key = parts[0]?.trim();
      if (!key) continue;
      let val: ReturnType<typeof JSON.parse> = parts.slice(1).join(':').trim();

      if (val.startsWith('[') && val.endsWith(']')) {
        val = val
          .substring(1, val.length - 1)
          .split(',')
          .map((s: string) => s.trim().replace(/['"]/g, ''));
      } else if (!Number.isNaN(Number(val))) {
        val = parseFloat(val);
      } else {
        val = val.replace(/['"]/g, '');
      }
      currentMod.options[key] = val;
    }
  }
  return modifiers;
}

/**
 * Executes `applyRecipe`.
 * @param graph Parameter `graph`.
 * @param recipeYaml Parameter `recipeYaml`.
 */
export function applyRecipe(graph: Graph, recipeYaml: string): void {
  const modifiers = parseRecipe(recipeYaml);
  for (const mod of modifiers) {
    mod.apply(graph);
  }
  graph.metadataProps.onnx9000_sparse_recipe = recipeYaml;
}
