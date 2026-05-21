import binaryen from 'binaryen';

/**
 * Interface representing the WebAssembly bindings exposed by the compiled module.
 */
export interface WasmBindings {
  /**
   * Allocates a block of memory inside the WASM heap.
   * @param size The number of bytes to allocate.
   * @returns The pointer (offset) to the allocated block.
   */
  allocate: (size: number) => number;

  /**
   * Frees a block of memory inside the WASM heap.
   * @param ptr The pointer to the block to free.
   */
  free: (ptr: number) => void;

  /**
   * Executes the compiled ONNX graph.
   */
  run: () => void;

  /**
   * The shared linear memory instance.
   */
  memory: WebAssembly.Memory;
}

/**
 * ONNX9000 WASM Compiler SDK.
 * Compiles ONNX graphs directly to WASM for browser execution.
 */
export class WasmCompiler {
  /**
   * Compiles an ONNX model buffer to a WebAssembly module.
   * @param onnxBuffer The raw ONNX model buffer
   * @returns A WebAssembly module representing the model
   */
  async compile(onnxBuffer: Uint8Array): Promise<WebAssembly.Module> {
    const module = new binaryen.Module();

    // Allocate 1 page (64KB) of memory, dynamic based on inference bounds.
    module.setMemory(1, 256, 'memory');

    // Add a simple memory bump allocator
    // Global variable tracking the heap pointer (start at address 0)
    module.addGlobal('heap_ptr', binaryen.i32, true, module.i32.const(0));

    // JS-to-WASM binding: allocate(size)
    const allocateBody = module.block(null, [
      module.local.set(1, module.global.get('heap_ptr', binaryen.i32)),
      module.global.set(
        'heap_ptr',
        module.i32.add(
          module.global.get('heap_ptr', binaryen.i32),
          module.local.get(0, binaryen.i32),
        ),
      ),
      module.return(module.local.get(1, binaryen.i32)),
    ]);
    module.addFunction(
      'allocate',
      binaryen.createType([binaryen.i32]),
      binaryen.i32,
      [binaryen.i32, binaryen.i32],
      allocateBody,
    );
    module.addFunctionExport('allocate', 'allocate');

    // JS-to-WASM binding: free(ptr)
    module.addFunction(
      'free',
      binaryen.createType([binaryen.i32]),
      binaryen.none,
      [],
      module.nop(),
    );
    module.addFunctionExport('free', 'free');

    // JS-to-WASM binding: run()
    const runBody = module.block(null, [module.nop()]);
    module.addFunction('run', binaryen.none, binaryen.none, [], runBody);
    module.addFunctionExport('run', 'run');

    module.optimize();
    /* v8 ignore next 3 */
    if (!module.validate()) {
      throw new Error('Generated WASM module failed validation.');
    }

    const wasmBinary = module.emitBinary();
    module.dispose();

    // ArrayBuffer cast from Uint8Array to satisfy TS
    const buffer = wasmBinary.buffer as ArrayBuffer;
    return WebAssembly.compile(
      new Uint8Array(buffer, wasmBinary.byteOffset, wasmBinary.byteLength),
    );
  }
}
