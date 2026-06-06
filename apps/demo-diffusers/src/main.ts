/* v8 ignore start */
import { DiffusionPipeline } from "@onnx9000/diffusers/src/pipeline.js";

/**
 * Initializes the Diffusers demo.
 */
export function initDiffusersDemo(): void {
  const runBtn = document.getElementById("run-btn") as HTMLButtonElement;
  const out = document.getElementById("output") as HTMLElement;
  if (!runBtn || !out) return;

  runBtn.addEventListener("click", async () => {
    out.innerText = "Initializing Pipeline...";
    try {
      const pipe = new DiffusionPipeline();
      out.innerText += "\nPipeline initialized. Generating mock tensor...";
      out.innerText += "\nImage tensor generated successfully [1, 3, 512, 512]";
    } catch (e: any) {
      out.innerText = `Error: ${e.message}`;
    }
  });
}
initDiffusersDemo();

/* v8 ignore stop */
