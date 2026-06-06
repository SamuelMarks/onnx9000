/* v8 ignore start */
/**
 * Initializes the MLIR lowering demo.
 */
export function initMlirLoweringDemo(): void {
  const lowerBtn = document.getElementById("lowerBtn") as HTMLButtonElement;
  const resetBtn = document.getElementById("resetBtn") as HTMLButtonElement;
  const outputDiv = document.getElementById("output") as HTMLDivElement;

  if (!lowerBtn || !resetBtn || !outputDiv) return;

  const delay = (ms: number) => new Promise((res) => setTimeout(res, ms));

  const stages = [
    {
      title: "1. ONNX to MHLO (High-Level Dialect)",
      code: "func.func @main() {}",
    },
    { title: "2. MHLO to Linalg (Structural Dialect)", code: "linalg.generic" },
    {
      title: "3. Bufferization (Value -> Memory Semantics)",
      code: "memref.alloc",
    },
    {
      title: "4. Linalg to HAL & VM (Bytecode Generation)",
      code: "hal.command_buffer",
    },
    { title: "5. Standalone WebGPU WGSL Payload Generated", code: "@compute" },
  ];

  lowerBtn.addEventListener("click", async () => {
    lowerBtn.disabled = true;
    outputDiv.innerHTML = "";

    for (const stage of stages) {
      const stepDiv = document.createElement("div");
      stepDiv.className = "step";

      const titleDiv = document.createElement("div");
      titleDiv.className = "step-title";
      titleDiv.textContent = stage.title;

      const codePre = document.createElement("pre");
      codePre.textContent = stage.code;

      stepDiv.appendChild(titleDiv);
      stepDiv.appendChild(codePre);

      outputDiv.appendChild(stepDiv);
      outputDiv.scrollTop = outputDiv.scrollHeight;

      await delay(600);
    }

    const completeDiv = document.createElement("div");
    completeDiv.style.color = "#28a745";
    completeDiv.style.fontWeight = "bold";
    completeDiv.textContent = "MLIR Lowering Pipeline Completed Successfully!";
    outputDiv.appendChild(completeDiv);
    outputDiv.scrollTop = outputDiv.scrollHeight;

    resetBtn.disabled = false;
  });

  resetBtn.addEventListener("click", () => {
    outputDiv.innerHTML =
      'Ready to compile. Click "Run MLIR Lowering Pass" to begin.';
    lowerBtn.disabled = false;
    resetBtn.disabled = true;
  });
}

// Attach listener for DOM content loaded if used in standard context
document.addEventListener("DOMContentLoaded", initMlirLoweringDemo);
// For testing environment where DOM is already loaded
if (
  document.readyState === "complete" ||
  document.readyState === "interactive"
) {
  initMlirLoweringDemo();
}

/* v8 ignore stop */
