import { Onnx9000Server } from './index';

// 25. Provide AWS Lambda native handler formats (`event, context`).
// 27. Gracefully catch specific runtime timeouts (e.g., Lambda 15min limit).
export function createLambdaHandler(server: Onnx9000Server) {
  return async function handler(event: any, context: any) {
    const method = event.httpMethod || event.requestContext?.http?.method || 'GET';
    const path = event.path || event.rawPath || '/';
    const query = new URLSearchParams(event.queryStringParameters || {}).toString();
    const url = `https://${event.headers?.host || 'localhost'}${path}${query ? '?' + query : ''}`;

    const headers = new Headers();
    if (event.headers) {
      for (const [key, value] of Object.entries(event.headers)) {
        if (typeof value === 'string') {
          headers.append(key, value);
        }
      }
    }

    let body: any = undefined;
    if (event.body) {
      if (event.isBase64Encoded) {
        body = Uint8Array.from(atob(event.body), (c) => c.charCodeAt(0));
      } else {
        body = event.body;
      }
    }

    const request = new Request(url, { method, headers, body });

    // Catch timeout
    const timeoutPromise = new Promise<Response>((_, reject) => {
      const remainingTime = context.getRemainingTimeInMillis
        ? context.getRemainingTimeInMillis()
        : 10000;
      // Timeout 100ms before actual Lambda timeout to respond gracefully
      setTimeout(
        () => reject(new Error('Lambda Timeout Reached')),
        Math.max(0, remainingTime - 100),
      );
    });

    try {
      const response = await Promise.race([server.fetch(request), timeoutPromise]);

      const responseHeaders: Record<string, string> = {};
      response.headers.forEach((value, key) => {
        responseHeaders[key] = value;
      });

      const responseBody = await response.text();

      return {
        statusCode: response.status,
        headers: responseHeaders,
        body: responseBody,
        isBase64Encoded: false,
      };
    } catch (err: any) {
      return {
        statusCode: 504,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ error: err.message }),
        isBase64Encoded: false,
      };
    }
  };
}
