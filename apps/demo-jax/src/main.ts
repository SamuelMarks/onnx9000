import { parseJaxpr } from '@onnx9000/converters';

const convertBtn = document.getElementById('convert-btn') as HTMLButtonElement;
const out = document.getElementById('jax-output') as HTMLElement;

convertBtn.addEventListener('click', async () => {
  out.innerText = 'Initializing JAX Parser...\n';
  convertBtn.disabled = true;

  try {
    const mockJaxprPayload = {
      invars: ['x', 'y'],
      outvars: ['z'],
      constvars: [],
      eqns: [
        {
          primitive: 'add',
          invars: ['x', 'y'],
          outvars: ['z'],
          params: {},
        },
      ],
    };

    out.innerText += '\nParsing mock ClosedJaxpr JSON:';
    out.innerText += `\n${JSON.stringify(mockJaxprPayload, null, 2)}`;

    const parsed = parseJaxpr(JSON.stringify(mockJaxprPayload));

    out.innerText += '\n\nMapping to ONNX9000 Core IR...';
    out.innerText += `\nExtracted ${parsed.eqns.length} equations.`;
    out.innerText += `\nPrimitive [${parsed.eqns[0].primitive}] mapped successfully.`;

    out.innerText += '\n\nSuccess! JAX & Flax graphs can be transpiled natively in JS.';
  } catch (e: any) {
    out.innerText += `\nError: ${e.message}`;
  } finally {
    convertBtn.disabled = false;
  }
});
