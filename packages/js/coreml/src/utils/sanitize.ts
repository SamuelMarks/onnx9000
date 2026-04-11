/* eslint-disable */
/**
 * CoreML serialization utility functions to maintain compatibility with Apple tooling.
 * @module
 */

/**
 * Sanitizes a string metadata entry by removing null terminators and invalid characters
 * which commonly disrupt JSON serialization inside an `.mlpackage` directory.
 * @param str - The raw string to sanitize.
 * @returns The cleanly formatted string, or undefined if the input was not a string.
 */
export function sanitizeMetadataString(str: string | undefined): string | undefined {
  if (typeof str !== 'string') return str;
  // Strip null terminators (\0) and ensure valid utf-8 strings for JSON serialization
  return str.replace(/\0/g, '').replace(/[\uFFFD\uFFFE\uFFFF]/g, '');
}

/**
 * Ensures generated filenames for weights inside an `.mlpackage` contain no illegal characters.
 * @param name - The original filename.
 * @returns A safe filename.
 */
export function sanitizeFilename(name: string): string {
  // Ensure generated filenames for weights inside .mlpackage contain no illegal characters
  return name.replace(/[^a-zA-Z0-9_\-\.]/g, '_');
}
