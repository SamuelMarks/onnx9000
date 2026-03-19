class TVMDemo {
  statusEl: HTMLElement;
  logEl: HTMLElement;
  compileBtn: HTMLButtonElement;
  runBtn: HTMLButtonElement;
  inputEl: HTMLTextAreaElement;

  wasmInstance: WebAssembly.Instance | null = null;

  constructor() {
    this.statusEl = document.getElementById('tvm-status') as HTMLElement;
    this.logEl = document.getElementById('tvm-output-log') as HTMLElement;
    this.compileBtn = document.getElementById('tvm-compile-btn') as HTMLButtonElement;
    this.runBtn = document.getElementById('tvm-run-btn') as HTMLButtonElement;
    this.inputEl = document.getElementById('tvm-input-model') as HTMLTextAreaElement;

    this.compileBtn.disabled = false;
    this.statusEl.innerText = 'Compiler Ready.';
    this.statusEl.className = 'status-badge status-ready';

    this.compileBtn.addEventListener('click', () => this.compile());
    this.runBtn.addEventListener('click', () => this.run());

    this.inputEl.value = `def @main(%x: Tensor[(10, 10), float32]) {
  %1 = multiply(%x, 2.0f);
  add(%1, 1.0f)
}`;
  }

  appendLog(text: string) {
    const msgEl = document.createElement('div');
    msgEl.className = `system-message`;
    msgEl.style.marginBottom = '0.5rem';
    msgEl.innerText = text;
    this.logEl.appendChild(msgEl);
    this.logEl.scrollTop = this.logEl.scrollHeight;
  }

  async compile() {
    this.compileBtn.disabled = true;
    this.runBtn.disabled = true;
    this.statusEl.innerText = 'Compiling to WASM...';
    this.statusEl.className = 'status-badge status-loading';
    this.logEl.innerHTML = '';

    this.appendLog('Parsing Relay IR...');
    await new Promise((r) => setTimeout(r, 400));
    this.appendLog('Applying target-specific optimizations (AutoTVM/MetaSchedule)...');
    await new Promise((r) => setTimeout(r, 600));
    this.appendLog('Lowering to Tensor IR (TIR)...');
    await new Promise((r) => setTimeout(r, 500));
    this.appendLog('Emitting WebAssembly (WASM) binary format...');
    await new Promise((r) => setTimeout(r, 400));

    // Tiny WASM module that adds 2 numbers.
    // (module (func (export "add") (param i32 i32) (result i32) local.get 0 local.get 1 i32.add))
    const wasmCode = new Uint8Array([
      0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f,
      0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x07, 0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00,
      0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b,
    ]);

    try {
      const module = await WebAssembly.compile(wasmCode);
      const instance = await WebAssembly.instantiate(module, {});
      this.wasmInstance = instance;
      this.appendLog(`WASM compiled successfully. Binary size: ${wasmCode.length} bytes.`);
      this.appendLog(`Exports: ${Object.keys(instance.exports).join(', ')}`);

      this.statusEl.innerText = 'AOT Compilation Complete';
      this.statusEl.className = 'status-badge status-ready';
      this.runBtn.disabled = false;
    } catch (e: any) {
      this.appendLog(`Compilation failed: ${e.message}`);
      this.statusEl.innerText = 'Compilation Failed';
      this.statusEl.className = 'status-badge status-loading'; // fallback if error missing
    } finally {
      this.compileBtn.disabled = false;
    }
  }

  async run() {
    if (!this.wasmInstance) return;
    this.runBtn.disabled = true;
    this.statusEl.innerText = 'Executing WASM kernel...';
    this.statusEl.className = 'status-badge status-generating';

    this.appendLog('Executing WASM kernel...');

    // simulate execution delay
    await new Promise((r) => setTimeout(r, 300));

    const addFn = this.wasmInstance.exports.add as (a: number, b: number) => number;
    const a = Math.floor(Math.random() * 100);
    const b = Math.floor(Math.random() * 100);
    const result = addFn(a, b);

    this.appendLog(`Execution complete! add(${a}, ${b}) = ${result}`);
    this.statusEl.innerText = 'Execution Complete';
    this.statusEl.className = 'status-badge status-ready';
    this.runBtn.disabled = false;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  new TVMDemo();
});
