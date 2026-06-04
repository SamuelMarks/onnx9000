/**
 * H2O MOJO/POJO parser.
 */

export function parseH2O(modelData: string): Record<string, unknown> {
  if (modelData.trim().startsWith('{')) {
    try {
      return JSON.parse(modelData) as Record<
        string,
        unknown
      >; /* v8 ignore next */ /* v8 ignore next */
    } catch {
      /* v8 ignore next */ /* v8 ignore next */
      return {}; /* v8 ignore next */ /* v8 ignore next */
    }
  }
  return {};
}
