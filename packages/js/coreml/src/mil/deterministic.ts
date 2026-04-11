/* eslint-disable */
import { Model } from '../schema.js';

export function assertDeterministicBuild(model: Model): void {
  // 294. Ensure full deterministic compilation
  // Sort metadata keys, ensure arrays are properly ordered.
  if (model.description?.metadata?.creatorDefined) {
    const keys = Object.keys(model.description.metadata.creatorDefined).sort();
    const sortedCreatorDefined: Record<string, string> = {};
    for (const k of keys) {
      sortedCreatorDefined[k] = model.description.metadata.creatorDefined[k]!;
    }
    model.description.metadata.creatorDefined = sortedCreatorDefined;
  }

  if (model.description?.input) {
    model.description.input.sort((a, b) => a.name.localeCompare(b.name));
  }

  if (model.description?.output) {
    model.description.output.sort((a, b) => a.name.localeCompare(b.name));
  }
}
