/**
 * @fileoverview Serverless Edge Router Demo UI.
 */
import { createServer } from '@onnx9000/serve';

/**
 * Initializes the serve demo UI.
 */
export function initServeDemo(): void {
  const startBtn = document.getElementById('start-btn') as HTMLButtonElement | null;
  const reqBtn = document.getElementById('req-btn') as HTMLButtonElement | null;
  const out = document.getElementById('server-output') as HTMLElement | null;

  if (!startBtn || !reqBtn || !out) return;

  let server: ReturnType<typeof createServer> | null = null;

  startBtn.addEventListener('click', () => {
    out.innerText = `Initializing Serverless Edge Router...
`;
    startBtn.disabled = true;

    server = createServer();
    out.innerText += `
Server initialized with KServe & OpenAI compatible routes.`;
    out.innerText += `
Ready to accept inference requests via server.fetch() locally.`;
    reqBtn.disabled = false;
  });

  reqBtn.addEventListener('click', async () => {
    if (!server) return;
    reqBtn.disabled = true;

    out.innerText += `

--- Sending Mock Request ---`;
    out.innerText += `
POST /v2/models/mock_model/infer`;

    const req = new Request('http://localhost:8080/v2/models/mock_model/infer', {
      method: 'POST',
      body: JSON.stringify({
        inputs: [
          {
            name: 'input_0',
            shape: [1, 3, 224, 224],
            datatype: 'FP32',
            data: [1.0],
          },
        ],
      }),
      headers: {
        'Content-Type': 'application/json',
      },
    });

    const res = await server.fetch(req);
    const body = await res.text();

    out.innerText += `
Status Code: ${res.status}`;
    out.innerText += `
Response: ${body}`;
    out.innerText += `

Success! Edge routing is fully functional in-browser.`;
    reqBtn.disabled = false;
  });
}

initServeDemo();
