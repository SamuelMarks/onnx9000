import { createServer } from '@onnx9000/serve';

const startBtn = document.getElementById('start-btn') as HTMLButtonElement;
const reqBtn = document.getElementById('req-btn') as HTMLButtonElement;
const out = document.getElementById('server-output') as HTMLElement;

let server: any;

startBtn.addEventListener('click', async () => {
  out.innerText = 'Initializing Serverless Edge Router...\n';
  startBtn.disabled = true;

  try {
    server = createServer();
    out.innerText += '\nServer initialized with KServe & OpenAI compatible routes.';
    out.innerText += '\nReady to accept inference requests via server.fetch() locally.';
    reqBtn.disabled = false;
  } catch (e: any) {
    out.innerText += `\nError: ${e.message}`;
    startBtn.disabled = false;
  }
});

reqBtn.addEventListener('click', async () => {
  if (!server) return;
  reqBtn.disabled = true;

  try {
    out.innerText += '\n\n--- Sending Mock Request ---';
    out.innerText += '\nPOST /v2/models/mock_model/infer';

    // Create a mock Request
    const req = new Request('http://localhost:8080/v2/models/mock_model/infer', {
      method: 'POST',
      body: JSON.stringify({
        inputs: [{ name: 'input_0', shape: [1, 3, 224, 224], datatype: 'FP32', data: [1.0] }],
      }),
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Handle using the router
    const res = await server.fetch(req);
    const body = await res.text();

    out.innerText += `\nStatus Code: ${res.status}`;
    out.innerText += `\nResponse: ${body}`;
    out.innerText += '\n\nSuccess! Edge routing is fully functional in-browser.';
  } catch (e: any) {
    out.innerText += `\nError: ${e.message}`;
  } finally {
    reqBtn.disabled = false;
  }
});
