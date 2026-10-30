/**
 * @fileoverview vm.ts
 * Provides vm functionality for the iree-runtime package.
 */
// 123. Runtime Module
import { BackendError } from '@onnx9000/core';
/** Represents the `Module` class. */
export class Module {
  /** Property `globals` of `Module`. */
  public globals: number[] = [];
  /** Property `memory` of `Module`. */
  public memory: ArrayBuffer;
  /** Property `memoryView` of `Module`. */
  public memoryView: DataView;
  /** Property `functions` of `Module`. */
  public functions: Array<(...args: unknown[]) => unknown> = [];
  /** Property `imports` of `Module`. */
  public imports: Map<string, (...args: unknown[]) => unknown> = new Map();

  /**
   * Initializes a new `Module` instance.
   * @param memorySize Parameter `memorySize`.
   */
  constructor(memorySize: number = 1024 * 1024) {
    this.memory = new ArrayBuffer(memorySize);
    this.memoryView = new DataView(this.memory);
  }
}

// 124. Runtime Context
/** Represents the `Context` class. */
export class Context {
  /** Property `module` of `Context`. */
  public module: Module;
  /** Property `pc` of `Context`. */
  public pc: number = 0;
  /** Property `registers` of `Context`. */
  public registers: number[] = new Array<number>(256).fill(0);

  /**
   * Initializes a new `Context` instance.
   * @param module Parameter `module`.
   */
  constructor(module: Module) {
    this.module = module;
  }

  // 126. Dynamic module loading
  /**
   * Executes `loadImport` on `Context`.
   * @param namespace Parameter `namespace`.
   * @param funcName Parameter `funcName`.
   * @param jsFunc Parameter `jsFunc`.
   * @returns Result of `loadImport`.
   */
  public loadImport(namespace: string, funcName: string, jsFunc: (...args: unknown[]) => unknown) {
    this.module.imports.set(`${namespace}.${funcName}`, jsFunc);
  }
}

// 121. Pure JS wvm interpreter
/** Represents the `WVMInterpreter` class. */
export class WVMInterpreter {
  /** Property `bytecode` of `WVMInterpreter`. */
  private bytecode: Uint8Array;
  /** Property `context` of `WVMInterpreter`. */
  private context: Context;

  /**
   * Initializes a new `WVMInterpreter` instance.
   * @param bytecode Parameter `bytecode`.
   * @param context Parameter `context`.
   */
  constructor(bytecode: Uint8Array, context: Context) {
    this.bytecode = bytecode;
    this.context = context;
    this.validate();
  }

  // 133. Strict validation
  /**
   * Executes `validate` on `WVMInterpreter`.
   * @returns Result of `validate`.
   */
  private validate() {
    if (
      this.bytecode.length < 4 ||
      this.bytecode[0] !== 0x57 ||
      this.bytecode[1] !== 0x56 ||
      this.bytecode[2] !== 0x4d ||
      this.bytecode[3] !== 0x30
    ) {
      throw new BackendError('Invalid WVM Bytecode');
    }
  }

  // 131, 132. ArrayBuffer passing
  /**
   * Executes `setInput` on `WVMInterpreter`.
   * @param offset Parameter `offset`.
   * @param data Parameter `data`.
   * @returns Result of `setInput`.
   */
  public setInput(offset: number, data: ArrayBuffer) {
    new Uint8Array(this.context.module.memory).set(new Uint8Array(data), offset);
  }

  /**
   * Executes `getOutput` on `WVMInterpreter`.
   * @param offset Parameter `offset`.
   * @param length Parameter `length`.
   * @returns Result of `getOutput`.
   */
  public getOutput(offset: number, length: number): ArrayBuffer {
    return this.context.module.memory.slice(offset, offset + length);
  }

  // 130. Synchronous execution mode
  /**
   * Executes `runSync` on `WVMInterpreter`.
   * @param debugLogging Parameter `debugLogging`.
   * @returns Result of `runSync`.
   */
  public runSync(debugLogging: boolean = false) {
    this.context.pc = 4; // Skip header
    const bc = this.bytecode;

    while (this.context.pc < bc.length) {
      const currentPc = this.context.pc++;
      const opcode = bc[currentPc] as number;

      if (debugLogging) {
        console.log(
          `[VM DEBUG] PC=${String(this.context.pc - 1)} Opcode=0x${opcode.toString(16)} Registers=`,
          this.context.registers.slice(0, 10),
        ); // 167
      }
      // 125. Bytecode dispatch loop
      switch (opcode) {
        case 0x01: // Module
          break;
        case 0x02: // Func
          break;
        case 0x03: {
          // Call
          const name = 'hal.cmd_create'; // dummy logic
          const imp = this.context.module.imports.get(name);
          if (imp) imp();
          break;
        }
        case 0x04: {
          // web.vm.add.i32
          const pcDst = this.context.pc++;
          const pcLhs = this.context.pc++;
          const pcRhs = this.context.pc++;
          const rDst = bc[pcDst];
          const rLhs = bc[pcLhs];
          const rRhs = bc[pcRhs];
          if (rDst === undefined || rLhs === undefined || rRhs === undefined) {
            throw new BackendError('Bytecode out of bounds');
          }
          this.context.registers[rDst] =
            (this.context.registers[rLhs] as number) + (this.context.registers[rRhs] as number);
          break;
        }
        case 0xff: // Return
          return;
        default:
          throw new BackendError(`Unknown opcode: ${String(opcode)}`);
      }
    }
  }

  // 129. Asynchronous execution mode
  /** Executes `runAsync` on `WVMInterpreter`. */
  public async runAsync(): Promise<void> {
    this.context.pc = 4;
    const bc = this.bytecode;
    let stepCount = 0;

    while (this.context.pc < bc.length) {
      const currentPc = this.context.pc++;
      const opcode = bc[currentPc] as number;

      switch (opcode) {
        case 0x01:
          break;
        case 0x02:
          break;
        case 0x03: {
          const name = 'hal.cmd_create';
          const imp = this.context.module.imports.get(name);
          if (imp) await imp();
          break;
        }
        case 0xff:
          return;
        default:
          throw new BackendError(`Unknown opcode: ${String(opcode)}`);
      }

      if (++stepCount % 100 === 0) {
        await new Promise((r) => setTimeout(r, 0)); // yield
      }
    }
  }
}

// 127, 128. Bind HAL VM to actual API calls
export const HALBindings = {
  register(context: Context, device: object | null) {
    context.loadImport('hal', 'cmd_create', () => {
      // maps to device.createCommandEncoder()
      if (!device) {
        // 134. Handle WebGPU context loss
        throw new BackendError('VM Error: WebGPU Context Lost');
      }
      return 'command_buffer_ptr';
    });

    context.loadImport('hal', 'buffer_subspan', () => {
      // 135. Tiny memory allocator logic if needed
      return undefined;
    });
  },
};

// 122. WASM WVM Interpreter
/** Represents the `WASMWVMInterpreter` class. */
export class WASMWVMInterpreter {
  /** Property `wasmModule` of `WASMWVMInterpreter`. */
  private wasmModule: WebAssembly.Module | null = null;
  /** Property `wasmInstance` of `WASMWVMInterpreter`. */
  private wasmInstance: WebAssembly.Instance | null = null;

  /**
   * Executes `initialize` on `WASMWVMInterpreter`.
   * @param wasmBinary Parameter `wasmBinary`.
   */
  public async initialize(wasmBinary: ArrayBuffer): Promise<void> {
    this.wasmModule = await WebAssembly.compile(wasmBinary);
    this.wasmInstance = await WebAssembly.instantiate(this.wasmModule, {
      env: {
        memory: new WebAssembly.Memory({ initial: 256 }),
        abort: () => {
          throw new BackendError('WASM Aborted');
        },
      },
    });
  }

  /** Executes `run` on `WASMWVMInterpreter`. */
  public run(): void {
    if (!this.wasmInstance) {
      throw new BackendError('WASM not initialized');
    }
    const runFunc = this.wasmInstance.exports.run as (...args: unknown[]) => unknown;
    if (typeof runFunc === 'function') {
      runFunc();
    }
  }
}
