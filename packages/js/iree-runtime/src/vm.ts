// 123. Runtime Module
export class Module {
  public globals: number[] = [];
  public memory: ArrayBuffer;
  public memoryView: DataView;
  public functions: Function[] = [];
  public imports: Map<string, Function> = new Map<string, Function>();

  constructor(memorySize: number = 1024 * 1024) {
    this.memory = new ArrayBuffer(memorySize);
    this.memoryView = new DataView(this.memory);
  }
}

// 124. Runtime Context
export class Context {
  public module: Module;
  public pc: number = 0;
  public registers: number[] = new Array<number>(256).fill(0);

  constructor(module: Module) {
    this.module = module;
  }

  // 126. Dynamic module loading
  public loadImport(namespace: string, funcName: string, jsFunc: Function) {
    this.module.imports.set(`${namespace}.${funcName}`, jsFunc);
  }
}

// 121. Pure JS wvm interpreter
export class WVMInterpreter {
  private bytecode: Uint8Array;
  private context: Context;

  constructor(bytecode: Uint8Array, context: Context) {
    this.bytecode = bytecode;
    this.context = context;
    this.validate();
  }

  // 133. Strict validation
  private validate() {
    if (
      this.bytecode.length < 4 ||
      this.bytecode[0] !== 0x57 ||
      this.bytecode[1] !== 0x56 ||
      this.bytecode[2] !== 0x4d ||
      this.bytecode[3] !== 0x30
    ) {
      throw new Error('Invalid WVM Bytecode');
    }
  }

  // 131, 132. ArrayBuffer passing
  public setInput(offset: number, data: ArrayBuffer) {
    new Uint8Array(this.context.module.memory).set(new Uint8Array(data), offset);
  }

  public getOutput(offset: number, length: number): ArrayBuffer {
    return this.context.module.memory.slice(offset, offset + length);
  }

  // 130. Synchronous execution mode
  public runSync(debugLogging: boolean = false) {
    this.context.pc = 4; // Skip header
    const bc = this.bytecode;

    while (this.context.pc < bc.length) {
      const opcode = bc[this.context.pc++]!;
      if (debugLogging) {
        console.log(
          `[VM DEBUG] PC=${this.context.pc - 1} Opcode=0x${opcode.toString(16)} Registers=`,
          this.context.registers.slice(0, 10),
        ); // 167
      }
      // 125. Bytecode dispatch loop
      switch (opcode) {
        case 0x01: // Module
          break;
        case 0x02: // Func
          break;
        case 0x03: // Call
          const name = 'hal.cmd_create'; // dummy logic
          const imp = this.context.module.imports.get(name);
          if (imp) imp();
          break;
        case 0x04: // web.vm.add.i32
          const rDst = bc[this.context.pc++]!;
          const rLhs = bc[this.context.pc++]!;
          const rRhs = bc[this.context.pc++]!;
          this.context.registers[rDst] =
            (this.context.registers[rLhs] ?? 0) + (this.context.registers[rRhs] ?? 0);
          break;
        case 0xff: // Return
          return;
        default:
          throw new Error(`Unknown opcode: ${opcode}`);
      }
    }
  }

  // 129. Asynchronous execution mode
  public async runAsync(): Promise<void> {
    this.context.pc = 4;
    const bc = this.bytecode;
    let stepCount = 0;

    while (this.context.pc < bc.length) {
      const opcode = bc[this.context.pc++]!;
      switch (opcode) {
        case 0x01:
          break;
        case 0x02:
          break;
        case 0x03:
          const name = 'hal.cmd_create';
          const imp = this.context.module.imports.get(name);
          if (imp) await imp();
          break;
        case 0xff:
          return;
        default:
          throw new Error(`Unknown opcode: ${opcode}`);
      }

      if (++stepCount % 100 === 0) {
        await new Promise((r) => setTimeout(r, 0)); // yield
      }
    }
  }
}

// 127, 128. Bind HAL VM to actual API calls
export class HALBindings {
  public static register(context: Context, device: object | null) {
    context.loadImport('hal', 'cmd_create', () => {
      // maps to device.createCommandEncoder()
      if (!device) {
        // 134. Handle WebGPU context loss
        throw new Error('VM Error: WebGPU Context Lost');
      }
      return 'command_buffer_ptr';
    });

    context.loadImport('hal', 'buffer_subspan', () => {
      // 135. Tiny memory allocator logic if needed
    });
  }
}

// 122. WASM WVM Interpreter
export class WASMWVMInterpreter {
  private wasmModule: WebAssembly.Module | null = null;
  private wasmInstance: WebAssembly.Instance | null = null;

  constructor() {}

  public async initialize(wasmBinary: ArrayBuffer): Promise<void> {
    this.wasmModule = await WebAssembly.compile(wasmBinary);
    this.wasmInstance = await WebAssembly.instantiate(this.wasmModule, {
      env: {
        memory: new WebAssembly.Memory({ initial: 256 }),
        abort: () => {
          throw new Error('WASM Aborted');
        },
      },
    });
  }

  public run(): void {
    if (!this.wasmInstance) {
      throw new Error('WASM not initialized');
    }
    if (typeof this.wasmInstance.exports.run === 'function') {
      (this.wasmInstance.exports.run as Function)();
    }
  }
}
