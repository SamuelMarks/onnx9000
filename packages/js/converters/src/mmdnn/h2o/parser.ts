/**
 * H2O MOJO/POJO parser.
 */

export function parseH2O(modelData: string): Record<string, unknown> {
  if (modelData.trim().startsWith('{')) {
    try {
      return JSON.parse(modelData) as Record<string, unknown>;
    } catch {
      return {};
    }
  }
  return {};
}
