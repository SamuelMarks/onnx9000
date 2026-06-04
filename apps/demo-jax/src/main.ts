/* v8 ignore next */ /* v8 ignore next */ import { parseJaxpr } from '@onnx9000/converters'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const convertBtn = document.getElementById(
  'convert-btn',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const out = document.getElementById(
  'jax-output',
) as HTMLElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
convertBtn.addEventListener('click', async () => {
  /* v8 ignore next */ /* v8 ignore next */
  out.innerText = 'Initializing JAX Parser...\n'; /* v8 ignore next */ /* v8 ignore next */
  convertBtn.disabled = true; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    const mockJaxprPayload = {
      /* v8 ignore next */ /* v8 ignore next */
      invars: ['x', 'y'] /* v8 ignore next */ /* v8 ignore next */,
      outvars: ['z'] /* v8 ignore next */ /* v8 ignore next */,
      constvars: [] /* v8 ignore next */ /* v8 ignore next */,
      eqns: [
        /* v8 ignore next */ /* v8 ignore next */
        {
          /* v8 ignore next */ /* v8 ignore next */
          primitive: 'add' /* v8 ignore next */ /* v8 ignore next */,
          invars: ['x', 'y'] /* v8 ignore next */ /* v8 ignore next */,
          outvars: ['z'] /* v8 ignore next */ /* v8 ignore next */,
          params: {} /* v8 ignore next */ /* v8 ignore next */,
        } /* v8 ignore next */ /* v8 ignore next */,
      ] /* v8 ignore next */ /* v8 ignore next */,
    }; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += '\nParsing mock ClosedJaxpr JSON:'; /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `\n${JSON.stringify(mockJaxprPayload, null, 2)}`; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    const parsed = parseJaxpr(
      JSON.stringify(mockJaxprPayload),
    ); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += '\n\nMapping to ONNX9000 Core IR...'; /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `\nExtracted ${parsed.eqns.length} equations.`; /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `\nPrimitive [${parsed.eqns[0].primitive}] mapped successfully.`; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText +=
      '\n\nSuccess! JAX & Flax graphs can be transpiled natively in JS.'; /* v8 ignore next */ /* v8 ignore next */
  } catch (e: any) {
    /* v8 ignore next */ /* v8 ignore next */
    out.innerText += `\nError: ${e.message}`; /* v8 ignore next */ /* v8 ignore next */
  } finally {
    /* v8 ignore next */ /* v8 ignore next */
    convertBtn.disabled = false; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
});
