/**
 * @fileoverview browser-io.ts
 * Provides browser-io functionality for the converters package.
 */
import { ConversionError } from '@onnx9000/core';
/**
 * Executes `readBrowserFile`.
 * @param file Parameter `file`.
 * @returns Result of `readBrowserFile`.
 */
export async function readBrowserFile(file: File | Blob): Promise<ArrayBuffer> {
  return file.arrayBuffer();
}

/**
 * Executes `fetchRemoteUrl`.
 * @param url Parameter `url`.
 * @param init Parameter `init`.
 * @returns Result of `fetchRemoteUrl`.
 */
export async function fetchRemoteUrl(url: string | URL, init?: RequestInit): Promise<ArrayBuffer> {
  const targetUrl = typeof url === 'string' ? url : url.href;
  const fetchOptions: RequestInit = {
    mode: 'cors',
    ...init,
  };

  const response = await fetch(targetUrl, fetchOptions);
  if (!response.ok) {
    throw new ConversionError(`Failed to fetch remote model: ${response.statusText}`);
  }

  return response.arrayBuffer();
}
