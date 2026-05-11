/**
 * Flax nnx state dict parser.
 */

export function parseFlaxState(content: string): Record<string, object> {
  const data = JSON.parse(content) as Record<string, object>;
  return data;
}
