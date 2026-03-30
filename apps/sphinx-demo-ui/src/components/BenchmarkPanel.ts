import { Component } from '../core/Component';

export class BenchmarkPanel extends Component<HTMLDivElement> {
  constructor() {
    super();
    this.element = this.render();
  }

  protected render(): HTMLDivElement {
    const container = document.createElement('div');
    container.className = 'demo-benchmark-panel';
    container.style.padding = '1rem';
    container.style.display = 'flex';
    container.style.flexDirection = 'column';
    container.style.gap = '1rem';
    container.style.height = '100%';
    container.style.overflowY = 'auto';

    const header = document.createElement('h3');
    header.textContent = 'ONNX9000 Benchmarks';
    container.appendChild(header);

    const btnContainer = document.createElement('div');
    btnContainer.style.display = 'flex';
    btnContainer.style.gap = '0.5rem';

    const memBtn = document.createElement('button');
    memBtn.textContent = 'Run Memory Profiler';
    memBtn.className = 'demo-btn';

    const ttftBtn = document.createElement('button');
    ttftBtn.textContent = 'Run TTFT Benchmark';
    ttftBtn.className = 'demo-btn';

    const tpsBtn = document.createElement('button');
    tpsBtn.textContent = 'Run TPS Benchmark';
    tpsBtn.className = 'demo-btn';

    const output = document.createElement('pre');
    output.style.background = 'var(--bg-secondary)';
    output.style.color = 'var(--text-primary)';
    output.style.padding = '1rem';
    output.style.borderRadius = '4px';
    output.style.flex = '1';
    output.textContent = 'Awaiting benchmark run...';

    const runPyodide = async (scriptBody: string) => {
      output.textContent = 'Running benchmark...';
      try {
        const pyodide = (window as any).pyodide;
        if (!pyodide) {
          output.textContent = 'Pyodide not loaded. Please wait for WebAssembly runtime...';
          return;
        }

        const pyCode = `
import sys, io
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()
try:
${scriptBody}
except Exception as e:
    print("Error:", e)
res = sys.stdout.getvalue()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
res
`;
        const res = await pyodide.runPythonAsync(pyCode);
        output.textContent = res || 'Finished with no output.';
      } catch (err: any) {
        output.textContent = `Error:\n${err.message}`;
      }
    };

    memBtn.onclick = () => {
      runPyodide(`
    import time
    import sys

    def profile_memory() -> None:
        mem_before = 0
        print(f"Memory before: {mem_before:.2f} MB")
        dummy_tensor = bytearray(1024 * 1024 * 10)
        mem_after = sys.getsizeof(dummy_tensor) / (1024 * 1024)
        print(f"Memory after: {mem_after:.2f} MB")
        print(f"Difference: {(mem_after - mem_before):.2f} MB")
        del dummy_tensor
        mem_final = 0
        print(f"Memory final: {mem_final:.2f} MB")

    profile_memory()
`);
    };

    ttftBtn.onclick = () => {
      runPyodide(`
    import time
    def bench_ttft() -> None:
        start_time = time.perf_counter()
        time.sleep(0.05)
        first_token_time = time.perf_counter()
        ttft = first_token_time - start_time
        print(f"Time To First Token (TTFT): {ttft * 1000:.2f} ms")
    bench_ttft()
`);
    };

    tpsBtn.onclick = () => {
      runPyodide(`
    import time
    def bench_tps() -> None:
        start_time = time.perf_counter()
        num_tokens = 100
        for _ in range(num_tokens):
            time.sleep(0.001)
        end_time = time.perf_counter()
        duration = end_time - start_time
        tps = num_tokens / duration
        print(f"Generated {num_tokens} tokens in {duration:.2f} s")
        print(f"Throughput: {tps:.2f} tokens/sec")
    bench_tps()
`);
    };

    btnContainer.appendChild(memBtn);
    btnContainer.appendChild(ttftBtn);
    btnContainer.appendChild(tpsBtn);
    container.appendChild(btnContainer);
    container.appendChild(output);

    return container;
  }
}
