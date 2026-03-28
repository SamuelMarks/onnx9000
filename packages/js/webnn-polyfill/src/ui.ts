export class PolyfillUI {
  private container: HTMLDivElement;

  constructor() {
    this.container = document.createElement('div');
    this.container.id = 'webnn-polyfill-ui-container';
    this.container.style.position = 'fixed';
    this.container.style.bottom = '10px';
    this.container.style.right = '10px';
    this.container.style.zIndex = '999999';
    this.container.style.fontFamily = 'monospace';
    this.container.style.fontSize = '12px';
    this.container.style.pointerEvents = 'none'; // Don't block clicks by default
    if (typeof document !== 'undefined') {
      document.body.appendChild(this.container);
    }
  }

  // 256. Provide visual feedback (spinners/bars) during long I/O operations natively.
  showSpinner(message: string) {
    const el = document.createElement('div');
    el.style.background = 'rgba(0,0,0,0.8)';
    el.style.color = '#0f0';
    el.style.padding = '8px';
    el.style.borderRadius = '4px';
    el.style.marginBottom = '4px';
    el.textContent = `[WebNN] ⌛ ${message}`;
    this.container.appendChild(el);
    return () => {
      if (this.container.contains(el)) {
        this.container.removeChild(el);
      }
    };
  }

  // 261. Expose interactive HTML Flamegraphs highlighting operations.
  showFlamegraph(profileData: any) {
    const el = document.createElement('div');
    el.style.background = '#222';
    el.style.color = '#fff';
    el.style.padding = '8px';
    el.style.border = '1px solid #444';
    el.style.pointerEvents = 'auto'; // allow interaction
    el.innerHTML = `<strong>WebNN Flamegraph</strong><br/>`;

    // Very simplistic flamegraph
    for (const op of profileData) {
      const bar = document.createElement('div');
      bar.style.height = '12px';
      bar.style.background = 'orange';
      bar.style.width = `${Math.min(100, op.time * 10)}%`;
      bar.style.margin = '2px 0';
      bar.title = `${op.name}: ${op.time}ms`;
      el.appendChild(bar);
    }

    this.container.appendChild(el);
    setTimeout(() => {
      if (this.container.contains(el)) {
        this.container.removeChild(el);
      }
    }, 5000);
  }

  // 277. Render graph connections dynamically in console UI.
  logGraphConsoleUI(graphInfo: any) {
    console.log('%c[WebNN Polyfill AST]', 'color: #bada55; font-weight: bold;');
    for (const node of graphInfo.nodes) {
      console.log(`  ├─ ${node.opType} (${node.name})`);
      console.log(`  │   ├─ Inputs: ${node.inputs.join(', ')}`);
      console.log(`  │   └─ Outputs: ${node.outputs.join(', ')}`);
    }
  }

  // 202. Expose a diagnostic flag on window.ML allowing developers to see the translated ONNX AST visually.
  showAST(graphInfo: any) {
    const el = document.createElement('div');
    el.style.background = '#111';
    el.style.color = '#0ff';
    el.style.padding = '12px';
    el.style.border = '2px solid #0ff';
    el.style.pointerEvents = 'auto';
    el.style.maxHeight = '300px';
    el.style.overflow = 'auto';
    el.innerHTML = `<strong>ONNX AST Visualizer</strong><br/><pre>${JSON.stringify(graphInfo, null, 2)}</pre>`;

    const closeBtn = document.createElement('button');
    closeBtn.textContent = 'Close';
    closeBtn.onclick = () => this.container.removeChild(el);
    el.appendChild(closeBtn);

    this.container.appendChild(el);
  }
}

export const polyfillUI = new PolyfillUI();
