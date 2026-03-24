/* prettier-ignore-start */
// ONNX9000 WebAssembly Execution Engine Scaffold

@external("env", "abort")
declare function abort(): void;

export function init(): void {
  // Setup logic for the engine
}

export function execute_graph(graph_ptr: usize, len: usize): i32 {
  // Mock execution return code
  return 0;
}
/* prettier-ignore-end */
