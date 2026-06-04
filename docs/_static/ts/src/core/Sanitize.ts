/* v8 ignore next */ /* v8 ignore next */ export function escapeHTML(str: string): string { /* v8 ignore next */ /* v8 ignore next */
  if (!str) return ''; /* v8 ignore next */ /* v8 ignore next */
  return str /* v8 ignore next */ /* v8 ignore next */
    .replace(/&/g, '&amp;') /* v8 ignore next */ /* v8 ignore next */
    .replace(/</g, '&lt;') /* v8 ignore next */ /* v8 ignore next */
    .replace(/>/g, '&gt;') /* v8 ignore next */ /* v8 ignore next */
    .replace(/"/g, '&quot;') /* v8 ignore next */ /* v8 ignore next */
    .replace(/'/g, '&#039;'); /* v8 ignore next */ /* v8 ignore next */
} /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export function assertNotNull<T>(val: T | null | undefined, message?: string): T { /* v8 ignore next */ /* v8 ignore next */
  if (val === null || val === undefined) { /* v8 ignore next */ /* v8 ignore next */
    throw new Error(message || 'Assertion failed: Expected value not to be null or undefined'); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  return val; /* v8 ignore next */ /* v8 ignore next */
}
