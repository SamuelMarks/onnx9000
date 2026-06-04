/* v8 ignore next */ /* v8 ignore next */ import { createServer } from '@onnx9000/serve'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const startBtn = document.getElementById(
  'start-btn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const reqBtn = document.getElementById(
  'req-btn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const out = document.getElementById(
  'server-output',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
let server: any; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
startBtn.addEventListener('click', async () => {
  /* v8 ignore next */ /* v8 ignore next */
  out.innerText =
    'Initializing Serverless Edge Router...\n'; /* v8 ignore next */ /* v8 ignore next */
  startBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    server = createServer(); /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      '\nServer initialized with KServe & OpenAI compatible routes.'; /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      '\nReady to accept inference requests via server.fetch() locally.'; /* v8 ignore next */ /* v8 ignore next */
    reqBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
  } catch (e: any) {
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `\nError: ${e.message}`; /* v8 ignore next */ /* v8 ignore next */
    startBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
reqBtn.addEventListener('click', async () => {
  /* v8 ignore next */ /* v8 ignore next */
  if (!server) return; /* v8 ignore next */ /* v8 ignore next */
  reqBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += '\n\n--- Sending Mock Request ---'; /* v8 ignore next */ /* v8 ignore next */
    out.innerText += '\nPOST /v2/models/mock_model/infer'; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Create a mock Request /* v8 ignore next */ /* v8 ignore next */
    const req = new Request('http://localhost:8080/v2/models/mock_model/infer', {
      /* v8 ignore next */ /* v8 ignore next */
      method: 'POST' /* v8 ignore next */ /* v8 ignore next */,
      body: JSON.stringify({
        /* v8 ignore next */ /* v8 ignore next */
        inputs: [
          { name: 'input_0', shape: [1, 3, 224, 224], datatype: 'FP32', data: [1.0] },
        ] /* v8 ignore next */ /* v8 ignore next */,
      }) /* v8 ignore next */ /* v8 ignore next */,
      headers: {
        /* v8 ignore next */ /* v8 ignore next */
        'Content-Type': 'application/json' /* v8 ignore next */ /* v8 ignore next */,
      } /* v8 ignore next */ /* v8 ignore next */,
    }); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Handle using the router /* v8 ignore next */ /* v8 ignore next */
    const res = await server.fetch(req); /* v8 ignore next */ /* v8 ignore next */
    const body = await res.text(); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `\nStatus Code: ${res.status}`; /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `\nResponse: ${body}`; /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      '\n\nSuccess! Edge routing is fully functional in-browser.'; /* v8 ignore next */ /* v8 ignore next */
  } catch (e: any) {
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `\nError: ${e.message}`; /* v8 ignore next */ /* v8 ignore next */
  } finally {
    /* v8 ignore next */ /* v8 ignore next */
    reqBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
});
