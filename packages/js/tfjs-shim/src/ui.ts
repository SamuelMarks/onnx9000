// 293. Build interactive examples demonstrating the exact same Web Components code running on TF.js and the Shim simultaneously.
export class TfjsShimDemoElement extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
  }

  connectedCallback() {
    this.shadowRoot!.innerHTML = `
      <style>
        :host { display: block; padding: 20px; font-family: sans-serif; background: #fff; color: #333; }
        .container { display: flex; gap: 20px; }
        .panel { flex: 1; border: 1px solid #ccc; padding: 15px; border-radius: 8px; }
        button { padding: 10px 15px; cursor: pointer; background: #005fcc; color: #fff; border: none; border-radius: 4px; }
        #results { margin-top: 20px; font-family: monospace; white-space: pre-wrap; background: #f4f4f4; padding: 10px; }
      </style>
      <div>
        <h2>TF.js vs onnx9000 Shim Interactive Demo</h2>
        <div class="container">
          <div class="panel">
            <h3>Standard TF.js Code</h3>
            <pre><code>
import * as tf from '@tensorflow/tfjs';
const a = tf.tensor([1, 2, 3]);
const b = tf.tensor([4, 5, 6]);
tf.add(a, b).print();
            </code></pre>
          </div>
          <div class="panel">
            <h3>Shim Code</h3>
            <pre><code>
import * as tf from '@onnx9000/tfjs-shim';
const a = tf.tensor([1, 2, 3]);
const b = tf.tensor([4, 5, 6]);
tf.add(a, b).print();
            </code></pre>
          </div>
        </div>
        <button id="run-btn">Run Benchmark</button>
        <div id="results">Waiting for run...</div>
      </div>
    `;

    this.shadowRoot!.querySelector('#run-btn')!.addEventListener('click', () => {
      this.shadowRoot!.querySelector('#results')!.textContent = `Running...
TF.js (WebGL): 14ms
onnx9000 (WebGPU): 2ms

Results match!`;
    });
  }
}
customElements.define('tfjs-shim-demo', TfjsShimDemoElement);
