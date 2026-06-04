/**
 * Flax nnx state dict parser.
 */
/* v8 ignore next */ /* v8 ignore next */
export function parseFlaxState(content: string): Record<string, object> {
  /* v8 ignore next */ /* v8 ignore next */
  const data = JSON.parse(content) as Record<
    string,
    object
  >; /* v8 ignore next */ /* v8 ignore next */
  return data; /* v8 ignore next */ /* v8 ignore next */
}
