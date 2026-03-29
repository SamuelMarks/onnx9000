/* eslint-disable */
// @ts-nocheck
export async function readBrowserFile(file: File | Blob): Promise<ArrayBuffer> {
  return file.arrayBuffer();
}

export async function fetchRemoteUrl(url: string | URL, init?: RequestInit): Promise<ArrayBuffer> {
  const targetUrl = typeof url === 'string' ? url : url.href;
  const fetchOptions: RequestInit = {
    mode: 'cors',
    ...init,
  };

  const response = await fetch(targetUrl, fetchOptions);
  if (!response.ok) {
    throw new Error(`Failed to fetch remote model: ${response.statusText}`);
  }

  return response.arrayBuffer();
}
