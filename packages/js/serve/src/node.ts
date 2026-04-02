import * as http from 'node:http';
import * as http2 from 'node:http2';
import { Onnx9000Server } from './index';

// 21. Provide `Node.js` specific entrypoint (`http` wrapper)
// 4. Implement HTTP/2 multiplexed connections.
// 5. Implement gRPC protocol emulation over HTTP/2 natively in JS.
export function serveNode(server: Onnx9000Server, port: number = 8080, useHttp2: boolean = false) {
  const handler = async (req: any, res: any) => {
    try {
      // Reconstruct full URL
      const protocol = req.socket.encrypted ? 'https' : 'http';
      const host = req.headers[':authority'] || req.headers.host || 'localhost';
      const urlStr = req.url || '/';
      const url = new URL(urlStr, `${protocol}://${host}`);

      // Reconstruct Request Headers
      const headers = new Headers();
      for (const [key, value] of Object.entries(req.headers)) {
        if (key.startsWith(':')) continue; // skip http2 pseudo-headers
        if (Array.isArray(value)) {
          value.forEach((v) => {
            headers.append(key, v);
          });
        } else if (value) {
          headers.append(key, value as string);
        }
      }

      // Read Body
      const buffers: Buffer[] = [];
      for await (const chunk of req) {
        buffers.push(chunk);
      }
      const bodyBuffer = Buffer.concat(buffers);
      const init: RequestInit = {
        method: req.method || req.headers[':method'] || 'GET',
        headers,
      };

      if (init.method !== 'GET' && init.method !== 'HEAD') {
        init.body = bodyBuffer;
      }

      const request = new Request(url, init);
      const response = await server.fetch(request);

      // Map back to Response
      if (res.stream) {
        // HTTP/2
        const responseHeaders: Record<string, string | number> = {
          ':status': response.status,
        };
        response.headers.forEach((value, key) => {
          responseHeaders[key] = value;
        });
        res.stream.respond(responseHeaders);
      } else {
        // HTTP/1
        res.statusCode = response.status;
        response.headers.forEach((value, key) => {
          res.setHeader(key, value);
        });
      }

      if (response.body) {
        const reader = response.body.getReader();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          res.write(value);
        }
        res.end();
      } else {
        res.end();
      }
    } catch (err: any) {
      if (res.stream) {
        res.stream.respond({ ':status': 500 });
      } else {
        res.statusCode = 500;
      }
      res.end(JSON.stringify({ error: err.message }));
    }
  };

  const httpServer = useHttp2
    ? http2.createServer(handler as any)
    : http.createServer(handler as any);

  httpServer.listen(port, () => {
    console.log(
      `ONNX9000 Serve listening on port ${port} (Node.js ${useHttp2 ? 'HTTP/2' : 'HTTP/1.1'})`,
    );
  });

  return httpServer;
}
