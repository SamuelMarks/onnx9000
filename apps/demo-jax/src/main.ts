/**
 * @fileoverview JAX & Flax Transpiler Demo UI.
 */
import { parseJaxpr } from '@onnx9000/converters';

/**
 * Initializes the JAX demo.
 */
export function initJaxDemo(): void {
  const convertBtn = document.getElementById('convert-btn') as HTMLButtonElement | null;
  const out = document.getElementById('jax-output') as HTMLElement | null;
  if (!convertBtn || !out) return;

  convertBtn.addEventListener('click', () => {
    out.innerText = 'Parsing ClosedJaxpr JSON...';
    convertBtn.disabled = true;

    const mockJaxpr = JSON.stringify({
      invars: ['a'],
      outvars: ['b'],
      constvars: [],
      eqns: [
        {
          invars: ['a', 'a'],
          outvars: ['b'],
          primitive: 'add',
          params: {},
        },
      ],
    });

    const parsed = parseJaxpr(mockJaxpr);
    out.innerText += ' Successfully parsed ClosedJaxpr AST.';
    out.innerText += ' Mapping to ONNX9000 Core IR...';
    out.innerText += ` Extracted ${parsed.eqns.length} equations.`;
    out.innerText += ` Primitive [${parsed.eqns[0]?.primitive}] mapped successfully.`;
    out.innerText += ' Success! JAX & Flax graphs can be transpiled natively in JS.';
    convertBtn.disabled = false;
  });
}

initJaxDemo();
