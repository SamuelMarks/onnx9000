import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { Attribute, Graph, Node, Onnx9000Error, Tensor, ValueInfo } from '@onnx9000/core';
import { describe, expect, it, vi } from 'vitest';
import * as ireeCompilerIndex from '../../iree-compiler/src/index.js';
import { MLIRInterop } from '../../iree-compiler/src/passes/interop.js';
import * as ireeRuntimeIndex from '../../iree-runtime/src/index.js';
import {
  Context,
  HALBindings,
  Module,
  WASMWVMInterpreter,
  WVMInterpreter,
} from '../../iree-runtime/src/vm.js';
import { extractMetadata, inferArchitecture } from '../../onnx2gguf/src/arch.js';
import { GGUFTensorType, GGUFValueType, GGUFWriter } from '../../onnx2gguf/src/builder.js';
import { compileGGUF } from '../../onnx2gguf/src/compiler.js';
import { fetchHfConfig, generateReadme } from '../../onnx2gguf/src/hub.js';
import { extractLlamaMetadata } from '../../onnx2gguf/src/llama.js';
import {
  f32ToF16,
  quantizeQ4_0,
  quantizeQ4_1,
  quantizeQ8_0,
} from '../../onnx2gguf/src/quantizer.js';
import { GGUFReader } from '../../onnx2gguf/src/reader.js';
import { reconstructONNX, reverseMapName, reverseMapType } from '../../onnx2gguf/src/reverse.js';
import { extractTokenizerMetadata } from '../../onnx2gguf/src/tokenizer.js';
import { LayoutOptimizer } from '../../tflite-exporter/src/compiler/layout.js';
import {
  ELEMENTWISE_OPS,
  mapConv2DOptions,
  mapDepthwiseConv2DOptions,
  mapOnnxNodeToTFLite,
  mapPool2DOptions,
  mapReducerOptions,
} from '../../tflite-exporter/src/compiler/operators.js';
import { compileGraphToTFLite } from '../../tflite-exporter/src/compiler/subgraph.js';
import { TFLiteExporter } from '../../tflite-exporter/src/exporter.js';
import { FlatBufferBuilder } from '../../tflite-exporter/src/flatbuffer/builder.js';
import { EdgeTPUOptimizer } from '../../tflite-exporter/src/optimizations/edgetpu.js';
import { TritonCompilerElement } from '../../triton-compiler/src/ui.js';
import { main as compilerCliMain, printHelp } from '../src/cli/index.js';
import { generateTriton } from '../src/triton/ast.js';
import { emitWGSL } from '../src/wgsl/index.js';

describe('Phase 6.3.5: Exhaustive 100% coverage for TS compilers & exporters', () => {
  it('covers @onnx9000/compiler cli, triton/ast, and wgsl', () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const exitSpy = vi.spyOn(process, 'exit').mockImplementation((() => {
      throw new Error('exit');
    }) as never);

    printHelp();
    const origArgv = process.argv;
    try {
      process.argv = ['node', 'cli.js'];
      expect(() => compilerCliMain()).toThrow('exit');

      process.argv = ['node', 'cli.js', '--help'];
      expect(() => compilerCliMain()).toThrow('exit');

      process.argv = ['node', 'cli.js', 'compile', '--flag-instead-of-model'];
      expect(() => compilerCliMain()).toThrow('exit');

      process.argv = ['node', 'cli.js', 'compile', '/non/existent/path/to/model.onnx'];
      expect(() => compilerCliMain()).toThrow('exit');

      const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'onnx9000-cli-'));
      const tmpModel = path.join(tmpDir, 'model.onnx');
      fs.writeFileSync(tmpModel, 'dummy');
      process.argv = ['node', 'cli.js', 'compile', tmpModel, '--boolean-flag', 'positional'];
      compilerCliMain();
      fs.rmSync(tmpDir, { recursive: true, force: true });
    } finally {
      process.argv = origArgv;
      logSpy.mockRestore();
      errSpy.mockRestore();
      exitSpy.mockRestore();
    }

    // triton/ast.ts branches
    const gEmptyName = new Graph('');
    gEmptyName.outputs.push('str_out' as never);
    const codeEmpty = generateTriton(gEmptyName);
    expect(codeEmpty).toContain('def fused_kernel(');

    const gTriton = new Graph('triton_branches');
    gTriton.inputs.push(new ValueInfo('x', [4], 'float32'));
    gTriton.outputs.push(new ValueInfo('y', [4], 'float32'));
    gTriton.nodes.push(new Node('Cast', ['x'], ['c1'], { to: 'float16' as never }));
    gTriton.nodes.push(new Node('Cast', ['x'], ['c2'], {}));
    gTriton.nodes.push(
      new Node('Cast', ['x'], ['c3'], { to: new Attribute('to', 'STRING', 'unknown_dtype') }),
    );
    gTriton.nodes.push(new Node('BitShift', ['x', 'x'], ['bs1'], { direction: 'RIGHT' as never }));
    gTriton.nodes.push(new Node('BitShift', ['x', 'x'], ['bs2'], {}));
    gTriton.nodes.push(
      new Node('BitShift', ['x', 'x'], ['bs3'], {
        direction: new Attribute('direction', 'STRING', 'LEFT'),
      }),
    );
    gTriton.nodes.push(new Node('Where', ['x', 'x'], ['w1'], {}));
    gTriton.nodes.push(new Node('Clip', ['x'], ['cl1'], {}));
    gTriton.nodes.push(new Node('LeakyRelu', ['x'], ['lr1'], { alpha: 0.2 as never }));
    gTriton.nodes.push(new Node('LeakyRelu', ['x'], ['lr2'], {}));
    gTriton.nodes.push(
      new Node('LeakyRelu', ['x'], ['lr3'], { alpha: new Attribute('alpha', 'FLOAT', 0.2) }),
    );
    gTriton.nodes.push(new Node('Constant', [], ['con1'], {}));
    gTriton.nodes.push(new Node('Add', [], [], {}));
    const codeBranches = generateTriton(gTriton);
    expect(codeBranches).toContain('tl.where');

    // wgsl/index.ts with ValueInfo object output
    const gWgsl = new Graph('wgsl_obj_out');
    gWgsl.inputs.push(new ValueInfo('a', [4], 'float32'));
    gWgsl.outputs.push(new ValueInfo('out_obj', [4], 'float32'));
    gWgsl.nodes.push(new Node('Relu', ['a'], ['out_obj']));
    expect(emitWGSL(gWgsl)).toContain('out_obj');
  });

  it('covers @onnx9000/iree-compiler and @onnx9000/iree-runtime', async () => {
    expect(ireeCompilerIndex).toBeDefined();
    expect(ireeRuntimeIndex).toBeDefined();

    const interop = new MLIRInterop();
    const reg = interop.parseMLIR(`web.lhs =   
%0 = web.add %a, %b`);
    expect(reg.blocks[0]?.operations.length).toBe(1);

    const mod = new Module();
    const ctx = new Context(mod);
    HALBindings.register(ctx, {});
    expect(mod.imports.get('hal.cmd_create')?.()).toBe('command_buffer_ptr');
    expect(mod.imports.get('hal.buffer_subspan')?.()).toBeUndefined();

    const ctxNullDev = new Context(new Module());
    HALBindings.register(ctxNullDev, null);
    expect(() => ctxNullDev.module.imports.get('hal.cmd_create')?.()).toThrow(Onnx9000Error);

    // WVMInterpreter: setInput, getOutput, debugLogging, opcode 0x01, 0x02, 0x03 (with and without import), 0x04 valid + out-of-bounds, unknown opcode, runAsync > 100 steps
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const ctxNoImp = new Context(new Module());
    const bcSync = new Uint8Array([
      0x57, 0x56, 0x4d, 0x30, 0x01, 0x02, 0x03, 0x04, 0x00, 0x01, 0x02, 0xff,
    ]);
    const interpSync = new WVMInterpreter(bcSync, ctxNoImp);
    interpSync.setInput(0, new Uint8Array([1, 2, 3, 4]).buffer);
    expect(new Uint8Array(interpSync.getOutput(0, 4))[0]).toBe(1);
    interpSync.runSync(true);
    const bcAsync = new Uint8Array([0x57, 0x56, 0x4d, 0x30, 0x01, 0x02, 0x03, 0xff]);
    await new WVMInterpreter(bcAsync, ctxNoImp).runAsync();
    logSpy.mockRestore();

    const bcOobAdd = new Uint8Array([0x57, 0x56, 0x4d, 0x30, 0x04, 0x00]);
    expect(() => new WVMInterpreter(bcOobAdd, ctx).runSync()).toThrow(Onnx9000Error);

    const bcUnknown = new Uint8Array([0x57, 0x56, 0x4d, 0x30, 0x99]);
    expect(() => new WVMInterpreter(bcUnknown, ctx).runSync()).toThrow(Onnx9000Error);
    await expect(new WVMInterpreter(bcUnknown, ctx).runAsync()).rejects.toThrow(Onnx9000Error);

    const longBytes = new Uint8Array(106);
    longBytes.set([0x57, 0x56, 0x4d, 0x30], 0);
    longBytes.fill(0x01, 4, 104);
    longBytes[104] = 0x02;
    longBytes[105] = 0xff;
    await new WVMInterpreter(longBytes, ctx).runAsync();

    // WASMWVMInterpreter
    const wasmInterp = new WASMWVMInterpreter();
    expect(() => wasmInterp.run()).toThrow(Onnx9000Error);

    const origCompile = WebAssembly.compile;
    const origInstantiate = WebAssembly.instantiate;
    let capturedAbort: (() => void) | undefined;
    let runCalled = false;
    WebAssembly.compile = vi.fn().mockResolvedValue({} as WebAssembly.Module);
    WebAssembly.instantiate = vi
      .fn()
      .mockImplementation((_mod: unknown, imports: { env: { abort: () => void } }) => {
        capturedAbort = imports.env.abort;
        return Promise.resolve({
          exports: {
            run: () => {
              runCalled = true;
            },
          },
        });
      });
    try {
      await wasmInterp.initialize(new ArrayBuffer(8));
      wasmInterp.run();
      expect(runCalled).toBe(true);
      expect(() => capturedAbort?.()).toThrow(Onnx9000Error);

      WebAssembly.instantiate = vi.fn().mockResolvedValue({ exports: {} });
      await wasmInterp.initialize(new ArrayBuffer(8));
      wasmInterp.run();
    } finally {
      WebAssembly.compile = origCompile;
      WebAssembly.instantiate = origInstantiate;
    }
  });

  it('covers @onnx9000/triton-compiler ui.ts', () => {
    const el = new TritonCompilerElement();
    // setCode before connectedCallback (when #output and #wgsl-output do not exist)
    el.setCode('before_connect', 'before_wgsl');
    el.connectedCallback();

    const dropzone = el.shadowRoot?.querySelector('#dropzone') as HTMLElement;
    dropzone.dispatchEvent(new Event('dragover'));
    expect(dropzone.classList.contains('hover')).toBe(true);
    dropzone.dispatchEvent(new Event('dragleave'));
    expect(dropzone.classList.contains('hover')).toBe(false);

    const dropEvtEmpty = new Event('drop');
    Object.assign(dropEvtEmpty, { dataTransfer: { files: [] } });
    dropzone.dispatchEvent(dropEvtEmpty);

    const dropEvtFile = new Event('drop');
    Object.assign(dropEvtFile, { dataTransfer: { files: [{ name: 'test.onnx' }] } });
    dropzone.dispatchEvent(dropEvtFile);
    expect(dropzone.innerHTML).toContain('test.onnx');

    const sliderM = el.shadowRoot?.querySelector('#blockM') as HTMLInputElement;
    sliderM.value = '64';
    sliderM.dispatchEvent(new Event('input'));
    expect(el.shadowRoot?.querySelector('#bm-val')?.textContent).toBe('64');

    const origCreateObjectURL = URL.createObjectURL;
    URL.createObjectURL = vi.fn().mockReturnValue('blob:test');
    try {
      const saveBtn = el.shadowRoot?.querySelector('#save') as HTMLButtonElement;
      saveBtn.click();
      el.setCode('print(1)', 'fn main() {}');
      el.setCode('print(2)');
      saveBtn.click();
    } finally {
      URL.createObjectURL = origCreateObjectURL;
    }

    const bundleStr = el.bundle({
      inputs: [
        { name: 'scalar_in', shape: [] },
        { name: 'vec_in', shape: [4] },
      ],
    });
    expect(bundleStr).toContain('scalar_in');
  });

  it('covers @onnx9000/onnx2gguf arch, builder, compiler, hub, llama, quantizer, reader, reverse, and tokenizer', async () => {
    // arch.ts
    for (const archName of [
      'mistral',
      'mixtral',
      'phi',
      'qwen',
      'gemma',
      'starcoder',
      'falcon',
      'bloom',
      'stablelm',
      'command-r',
      'bert',
    ]) {
      expect(inferArchitecture(new Graph(archName))).toBeTruthy();
    }
    const gUnk = new Graph('');
    expect(inferArchitecture(gUnk)).toBe('unknown');
    const gLlamaTensor = new Graph('');
    gLlamaTensor.tensors['llama.weight'] = new Tensor('llama.weight', [1], 'float32');
    expect(inferArchitecture(gLlamaTensor)).toBe('llama');
    expect(extractMetadata(gUnk)).toEqual({});
    expect(() => extractMetadata(gUnk, 'unsupported_arch')).toThrow(Onnx9000Error);
    expect(extractMetadata(gUnk, 'mistral')['mistral.attention.sliding_window']).toBe(4096);
    expect(extractMetadata(gUnk, 'gemma')['gemma.attention.layer_norm_rms_epsilon']).toBe(1e-6);
    expect(extractMetadata(gUnk, 'mixtral')['mixtral.expert_count']).toBe(8);

    // llama.ts
    const gLlama = new Graph('llama');
    gLlama.tensors['model.embed_tokens.weight'] = new Tensor(
      'model.embed_tokens.weight',
      [100, 64],
      'float32',
    );
    gLlama.tensors['model.layers.0.self_attn.q_proj.weight'] = new Tensor(
      'model.layers.0.self_attn.q_proj.weight',
      [64, 64],
      'float32',
    );
    gLlama.tensors['model.layers.0.self_attn.k_proj.weight'] = new Tensor(
      'model.layers.0.self_attn.k_proj.weight',
      [32, 64],
      'float32',
    );
    gLlama.tensors['model.layers.0.mlp.up_proj.weight'] = new Tensor(
      'model.layers.0.mlp.up_proj.weight',
      [256, 64],
      'float32',
    );
    gLlama.nodes.push(new Node('Silu', ['x'], ['y']));
    gLlama.nodes.push(
      new Node('RMSNormalization', ['y'], ['z'], {
        epsilon: new Attribute('epsilon', 'FLOAT', 1e-6),
      }),
    );
    const llamaMeta = extractLlamaMetadata(gLlama);
    expect(llamaMeta['custom.is_swiglu']).toBe(true);

    // builder.ts & reader.ts (all value types, error cases, and tensor types)
    const writer = new GGUFWriter();
    writer.addUint8('u8', 1);
    writer.addInt8('i8', -1);
    writer.addUint16('u16', 2);
    writer.addInt16('i16', -2);
    writer.addUint32('u32', 3);
    writer.addInt32('i32', -3);
    writer.addFloat32('f32', 1.5);
    writer.addUint64('u64', 4n);
    writer.addInt64('i64', -4n);
    writer.addFloat64('f64', 2.5);
    writer.addBool('b_true', true);
    writer.addBool('b_false', false);
    writer.addString('str', 'hello');
    writer.addArray('arr', [1, 2], GGUFValueType.INT32);
    writer.addTensorInfo('t_f32', [2n], GGUFTensorType.F32, 0n);
    writer.addTensorInfo('t_f16', [2n], GGUFTensorType.F16, 0n);
    writer.addTensorInfo('t_q40', [32n], GGUFTensorType.Q4_0, 0n);
    writer.addTensorInfo('t_q41', [32n], GGUFTensorType.Q4_1, 0n);
    writer.addTensorInfo('t_q80', [32n], GGUFTensorType.Q8_0, 0n);
    writer.addTensorInfo('t_unk', [1n], 999 as GGUFTensorType, 0n);

    const hSize = writer.getHeaderSize();
    const buf = new ArrayBuffer(hSize + 128);
    writer.writeHeader(buf);

    const reader = new GGUFReader(buf);
    expect(reader.kvs.u8).toBe(1);
    expect(reader.kvs.i8).toBe(-1);
    expect(reader.kvs.u16).toBe(2);
    expect(reader.kvs.i16).toBe(-2);
    expect(reader.kvs.u64).toBe(4n);
    expect(reader.kvs.i64).toBe(-4n);
    expect(reader.kvs.f64).toBe(2.5);
    expect(reader.kvs.b_true).toBe(true);
    expect(reader.kvs.b_false).toBe(false);
    expect(reader.getTensor('t_f32').byteLength).toBe(8);
    expect(reader.getTensor('t_f16').byteLength).toBe(4);
    expect(reader.getTensor('t_q40').byteLength).toBe(18);
    expect(reader.getTensor('t_q41').byteLength).toBe(20);
    expect(reader.getTensor('t_q80').byteLength).toBe(34);
    expect(() => reader.getTensor('missing')).toThrow(Onnx9000Error);
    expect(() => reader.getTensor('t_unk')).toThrow(Onnx9000Error);

    // Invalid value type in writer and reader
    const badWriter = new GGUFWriter();
    badWriter.kvs.push({ key: 'bad', val: 0, type: 999 as GGUFValueType });
    expect(() => badWriter.getHeaderSize()).toThrow(Onnx9000Error);
    const tmpWriter = new GGUFWriter();
    tmpWriter.addUint32('ok', 1);
    const badBuf = new ArrayBuffer(tmpWriter.getHeaderSize() + 64);
    (tmpWriter.kvs[0] as { type: GGUFValueType }).type = 999 as GGUFValueType;
    expect(() => tmpWriter.writeHeader(badBuf)).toThrow(Onnx9000Error);

    // Invalid GGUF magic, version, and value type in GGUFReader
    const notGguf = new ArrayBuffer(32);
    expect(() => new GGUFReader(notGguf)).toThrow(Onnx9000Error);
    const badVer = new ArrayBuffer(32);
    const bv = new DataView(badVer);
    bv.setUint8(0, 0x47);
    bv.setUint8(1, 0x47);
    bv.setUint8(2, 0x55);
    bv.setUint8(3, 0x46);
    bv.setUint32(4, 99, true);
    expect(() => new GGUFReader(badVer)).toThrow(Onnx9000Error);

    const badValBuf = new ArrayBuffer(64);
    const bvv = new DataView(badValBuf);
    bvv.setUint8(0, 0x47);
    bvv.setUint8(1, 0x47);
    bvv.setUint8(2, 0x55);
    bvv.setUint8(3, 0x46);
    bvv.setUint32(4, 3, true);
    bvv.setBigUint64(8, 0n, true);
    bvv.setBigUint64(16, 1n, true);
    bvv.setBigUint64(24, 1n, true);
    bvv.setUint8(32, 0x61);
    bvv.setUint32(33, 999, true);
    expect(() => new GGUFReader(badValBuf)).toThrow(Onnx9000Error);

    // compiler.ts branches (including tokMeta empty array so v[0] is neither number nor string)
    const gComp = new Graph('');
    (gComp as unknown as { doc_string: string }).doc_string = '  doc  ';
    gComp.initializers.push('missing_t', 'f16_t', 'dyn_t', 'unk_dtype_t');
    gComp.tensors.f16_t = new Tensor('f16_t', [2], 'float16', true, false, new Uint8Array(4));
    gComp.tensors.dyn_t = new Tensor('dyn_t', ['N'], 'float32', true, false, null);
    gComp.tensors.unk_dtype_t = new Tensor(
      'unk_dtype_t',
      [1],
      'unknown_dtype' as never,
      true,
      false,
      new Uint8Array(4),
    );
    const ggufOut = compileGGUF(gComp, {
      'general.alignment': 64,
      'tokenizer.json': JSON.stringify({ model: { vocab: {} } }),
      'custom.bool': true,
      'custom.str': 'val',
      'custom.f32_arr': [1.5, 2.5],
      'custom.str_arr': ['a', 'b'],
      'custom.bool_arr': [true, false],
      'custom.float': 1.25,
      'custom.int': 42,
      'custom.obj': {} as never,
    });
    expect(ggufOut.byteLength).toBeGreaterThan(0);

    // hub.ts branches: 404, 401, 500 (neither 404 nor 403/401), Error instance vs string rejection
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const origFetch = globalThis.fetch;
    try {
      globalThis.fetch = vi
        .fn()
        .mockResolvedValueOnce({ ok: false, status: 404 })
        .mockRejectedValueOnce('non-error rejection');
      await fetchHfConfig('org/model', 'tok');

      globalThis.fetch = vi
        .fn()
        .mockResolvedValueOnce({ ok: false, status: 401 })
        .mockRejectedValueOnce('non-error rejection');
      await fetchHfConfig('org/model', 'tok');

      globalThis.fetch = vi
        .fn()
        .mockResolvedValueOnce({ ok: false, status: 500 })
        .mockRejectedValueOnce(new Error('tok error'));
      await fetchHfConfig('org/model');

      globalThis.fetch = vi
        .fn()
        .mockRejectedValueOnce('non-error config rejection')
        .mockResolvedValueOnce({ ok: false, status: 500 });
      await fetchHfConfig('org/model');
    } finally {
      globalThis.fetch = origFetch;
      warnSpy.mockRestore();
    }
    expect(generateReadme('m', 'org/m', 'Q4_0')).toContain('Q4_0');

    // quantizer.ts branches (zero block + non-zero block, error branches)
    expect(() => quantizeQ4_0(new Uint8Array(4))).toThrow(Onnx9000Error);
    expect(() => quantizeQ4_1(new Uint8Array(4))).toThrow(Onnx9000Error);
    expect(() => quantizeQ8_0(new Uint8Array(4))).toThrow(Onnx9000Error);
    const f32Data = new Float32Array(64);
    for (let i = 32; i < 64; i++) f32Data[i] = (i - 48) * 0.5;
    const u8Data = new Uint8Array(f32Data.buffer);
    expect(f32ToF16(u8Data).byteLength).toBe(128);
    expect(quantizeQ4_0(u8Data).byteLength).toBe(36);
    expect(quantizeQ4_1(u8Data).byteLength).toBe(40);
    expect(quantizeQ8_0(u8Data).byteLength).toBe(68);

    // reverse.ts all tensor name mappings and types
    const blkNames = [
      'token_embd.weight',
      'blk.0.attn_norm.weight',
      'blk.0.attn_q.weight',
      'blk.0.attn_q.bias',
      'blk.0.attn_k.weight',
      'blk.0.attn_k.bias',
      'blk.0.attn_v.weight',
      'blk.0.attn_v.bias',
      'blk.0.attn_output.weight',
      'blk.0.attn_output.bias',
      'blk.0.attn_qkv.weight',
      'blk.0.ffn_norm.weight',
      'blk.0.ffn_gate.weight',
      'blk.0.ffn_down.weight',
      'blk.0.ffn_up.weight',
      'blk.0.ffn_gate_up.weight',
      'blk.0.ffn_gate_inp.weight',
      'blk.0.unknown.weight',
      'output_norm.weight',
      'output.weight',
      'unmapped.weight',
    ];
    for (const n of blkNames) {
      expect(reverseMapName(n)).toBeTruthy();
    }
    expect(reverseMapType(GGUFTensorType.F32)).toBe('float32');
    expect(reverseMapType(GGUFTensorType.F16)).toBe('float16');
    expect(reverseMapType(GGUFTensorType.Q4_0)).toBe('uint8');
    expect(reverseMapType(GGUFTensorType.Q4_1)).toBe('uint8');
    expect(reverseMapType(GGUFTensorType.Q8_0)).toBe('int8');
    expect(reverseMapType(999 as GGUFTensorType)).toBe('float32');

    const mockReader = {
      kvs: {
        'general.name': 'custom_recon_name',
        'general.architecture': 'llama',
        'tokenizer.ggml.tokens': ['a', 'b'],
        'split.index': 1,
      },
      tensors: {
        'blk.0.attn_q.weight': {
          name: 'blk.0.attn_q.weight',
          shape: [32n, 32n],
          type: GGUFTensorType.Q8_0,
          offset: 0n,
        },
      },
      getTensor: () => new Uint8Array(34 * 32),
    } as unknown as GGUFReader;
    const reconG = reconstructONNX(mockReader);
    expect(reconG.name).toBe('custom_recon_name');
    expect(reconstructONNX({ kvs: {}, tensors: {} } as unknown as GGUFReader).name).toBe(
      'reconstructed',
    );

    // tokenizer.ts branches (empty object {}, invalid JSON, Unigram, WordPiece, vocab pad/truncate, merges, added_tokens)
    expect(extractTokenizerMetadata('{}')['tokenizer.ggml.model']).toBe('gpt2');
    expect(extractTokenizerMetadata('{invalid_json')['tokenizer.ggml.model']).toBe('llama');
    expect(
      extractTokenizerMetadata(
        JSON.stringify({
          model: { type: 'Unigram', vocab: { a: 0 }, merges: ['a b'] },
          added_tokens: [{}],
        }),
        3,
      )['tokenizer.ggml.tokens'].length,
    ).toBe(3);
    expect(
      extractTokenizerMetadata(
        JSON.stringify({
          model: { type: 'WordPiece', vocab: { a: 0, b: 1, c: 2 } },
        }),
        2,
      )['tokenizer.ggml.tokens'].length,
    ).toBe(2);
    expect(
      extractTokenizerMetadata(JSON.stringify({ model: { vocab: [] } }))['tokenizer.ggml.tokens'],
    ).toBeUndefined();
  });

  it('covers @onnx9000/tflite-exporter operators, subgraph, edgetpu, and layout', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});

    // edgetpu.ts: Expand broadcast warning
    const gTpu = new Graph('tpu');
    gTpu.nodes.push(new Node('Expand', ['x', 'shape'], ['y'], {}, 'exp1'));
    expect(new EdgeTPUOptimizer(gTpu).optimize().some((w) => w.includes('Expanded 1'))).toBe(true);

    // operators.ts: exercise every createOptions both WITHOUT and WITH explicit attributes
    const fb = new FlatBufferBuilder();
    const gOp = new Graph('op_graph');
    gOp.tensors.shape_init = new Tensor(
      'shape_init',
      [2],
      'int32',
      true,
      false,
      new Int32Array([2, 3]),
    );
    const transNode = new Node(
      'Transpose',
      ['x0'],
      ['x_trans'],
      { perm: new Attribute('perm', 'INTS', [0, 2, 1]) },
      'tr_prod',
    );
    gOp.nodes.push(transNode);

    const richAttrs = {
      keepdims: new Attribute('keepdims', 'INT', 1),
      alpha: new Attribute('alpha', 'FLOAT', 0.2),
      beta: new Attribute('beta', 'FLOAT', 0.75),
      size: new Attribute('size', 'INT', 3),
      bias: new Attribute('bias', 'FLOAT', 2.0),
      begin_mask: new Attribute('begin_mask', 'INT', 1),
      end_mask: new Attribute('end_mask', 'INT', 1),
      shrink_axis_mask: new Attribute('shrink_axis_mask', 'INT', 1),
      ellipsis_mask: new Attribute('ellipsis_mask', 'INT', 1),
      new_axis_mask: new Attribute('new_axis_mask', 'INT', 1),
      axis: new Attribute('axis', 'INT', 1),
      adj_x: new Attribute('adj_x', 'INT', 1),
      adj_y: new Attribute('adj_y', 'INT', 1),
      blocksize: new Attribute('blocksize', 'INT', 2),
      exclusive: new Attribute('exclusive', 'INT', 1),
      reverse: new Attribute('reverse', 'INT', 1),
      strides: new Attribute('strides', 'INTS', [2, 2]),
      dilations: new Attribute('dilations', 'INTS', [2, 2]),
    };

    for (const [opName, mapping] of Object.entries(ELEMENTWISE_OPS)) {
      if (mapping.createOptions) {
        mapping.createOptions(
          fb,
          new Node(opName, ['x_trans', 'shape_init'], ['y1', 'y2'], {}),
          gOp,
        );
        mapping.createOptions(
          fb,
          new Node(opName, ['x_trans', 'shape_init'], ['y1', 'y2'], richAttrs),
          gOp,
        );
      }
    }
    // Cast all switch cases (1, 2, 3, 6, 7, 9, 10, 11)
    for (const toVal of [1, 2, 3, 6, 7, 9, 10, 11]) {
      ELEMENTWISE_OPS.Cast?.createOptions?.(
        fb,
        new Node('Cast', ['x'], ['y'], { to: new Attribute('to', 'INT', toVal) }),
      );
    }
    // Add/Sub/Mul/Div with Relu and Relu6
    for (const op of ['Add', 'Sub', 'Mul', 'Div']) {
      for (const act of ['Relu', 'Relu6']) {
        ELEMENTWISE_OPS[op]?.createOptions?.(
          fb,
          new Node(op, ['a', 'b'], ['c'], {
            fused_activation: new Attribute('fused_activation', 'STRING', act),
          }),
        );
      }
    }
    // Resize nearest + align_corners + half_pixel
    ELEMENTWISE_OPS.Resize?.createOptions?.(
      fb,
      new Node('Resize', ['x'], ['y'], {
        mode: new Attribute('mode', 'STRING', 'nearest'),
        coordinate_transformation_mode: new Attribute(
          'coordinate_transformation_mode',
          'STRING',
          'align_corners',
        ),
      }),
    );
    ELEMENTWISE_OPS.Resize?.createOptions?.(
      fb,
      new Node('Resize', ['x'], ['y'], {
        coordinate_transformation_mode: new Attribute(
          'coordinate_transformation_mode',
          'STRING',
          'half_pixel',
        ),
      }),
    );
    // Sequence RNNs with time_major = 1
    for (const op of [
      'UnidirectionalSequenceRNN',
      'UnidirectionalSequenceLSTM',
      'BidirectionalSequenceLSTM',
      'GRU',
    ]) {
      ELEMENTWISE_OPS[op]?.createOptions?.(
        fb,
        new Node(op, ['x'], ['y'], { time_major: new Attribute('time_major', 'INT', 1) }),
      );
    }
    // ConvTranspose, mapPool2DOptions, mapConv2DOptions, mapDepthwiseConv2DOptions with 1D, 2D, and empty attrs
    mapDepthwiseConv2DOptions(fb, new Node('Conv', ['x'], ['y'], {}));
    mapDepthwiseConv2DOptions(fb, new Node('Conv', ['x'], ['y'], richAttrs));
    for (const autoPad of ['SAME_UPPER', 'VALID', 'NOTSET']) {
      const attrs = {
        auto_pad: new Attribute('auto_pad', 'STRING', autoPad),
        strides: new Attribute('strides', 'INTS', [2]),
        dilations: new Attribute('dilations', 'INTS', [2]),
        kernel_shape: new Attribute('kernel_shape', 'INTS', [3]),
        pads: new Attribute('pads', 'INTS', [1, 1, 1, 1]),
        fused_activation: new Attribute(
          'fused_activation',
          'STRING',
          autoPad === 'SAME_UPPER' ? 'Relu' : 'Relu6',
        ),
      };
      const n = new Node('Conv', ['x'], ['y'], attrs);
      ELEMENTWISE_OPS.ConvTranspose?.createOptions?.(fb, n);
      mapPool2DOptions(fb, n);
      mapConv2DOptions(fb, n);
      mapDepthwiseConv2DOptions(fb, n);
    }
    mapReducerOptions(fb, new Node('ReduceMean', ['x'], ['y'], {}));
    mapReducerOptions(fb, new Node('ReduceMean', ['x'], ['y'], richAttrs));
    expect(
      mapOnnxNodeToTFLite(
        new Node('Resize', ['x'], ['y'], { mode: new Attribute('mode', 'STRING', 'nearest') }),
      )?.builtinCode,
    ).toBeDefined();
    expect(
      mapOnnxNodeToTFLite(new Node('NonMaxSuppression', ['x'], ['y'], {}))?.createOptions?.(
        fb,
        new Node('NonMaxSuppression', ['x'], ['y']),
      ),
    ).toBe(0);
    const customDomainNode = new Node('MyOp', ['x'], ['y']);
    customDomainNode.domain = 'custom.domain';
    expect(mapOnnxNodeToTFLite(customDomainNode)?.createOptions?.(fb, customDomainNode)).toBe(0);

    // subgraph.ts: Loop, If, Tokenizer, pytorch metadata, string tensor, Uint8Array tensor, Float32Array tensor, externalData, dynamic shape (-1), NonMaxSuppression, tf domain, custom_options, stripCustomOps
    const gSub = new Graph('');
    (gSub as unknown as { metadata: { producer_name: string } }).metadata = {
      producer_name: 'pytorch',
    };
    gSub.inputs.push(new ValueInfo('x_dyn', [-1, 4], 'float32'));
    gSub.outputs.push(new ValueInfo('y_out', [4], 'float32'));
    gSub.tensors.x_dyn = new Tensor('x_dyn', [-1, 4], 'float32');
    gSub.tensors.y_out = new Tensor('y_out', [4], 'float32');
    const strTensor = new Tensor('str_init', [2], 'string', true, false, null);
    (strTensor as unknown as { data: string[] }).data = ['hello', 'world'];
    gSub.tensors.str_init = strTensor;
    const strNonArr = new Tensor('str_non_arr', [1], 'string', true, false, new Uint8Array([65]));
    gSub.tensors.str_non_arr = strNonArr;
    gSub.tensors.u8_typed = new Tensor('u8_typed', [4], 'uint8', true, false, new Uint8Array(4));
    gSub.tensors.f32_typed = new Tensor(
      'f32_typed',
      [2],
      'float32',
      true,
      false,
      new Float32Array([1, 2]),
    );
    const extTensor = new Tensor('ext_t', [2], 'float32', true, false, null);
    extTensor.externalData = { location: 'weights.bin', offset: 0, length: 8 };
    gSub.tensors.ext_t = extTensor;

    gSub.nodes.push(new Node('Loop', ['x_dyn'], ['y_out'], {}, 'loop1'));
    gSub.nodes.push(new Node('If', ['x_dyn'], ['y_out'], {}, 'if1'));
    const tokNode = new Node('HFTokenizer', ['x_dyn'], ['y_out'], {}, 'tok1');
    tokNode.domain = 'ai.onnx.contrib';
    gSub.nodes.push(tokNode);
    const tfNode = new Node('TFSelect', ['x_dyn'], ['y_out'], {}, 'tf1');
    tfNode.domain = 'tf';
    gSub.nodes.push(tfNode);
    gSub.nodes.push(
      new Node(
        'NonMaxSuppression',
        ['x_dyn'],
        ['y_out'],
        { custom_options: new Attribute('custom_options', 'BYTES', new Uint8Array([1, 2, 3])) },
        'nms1',
      ),
    );
    gSub.nodes.push(new Node('FloorDiv', ['x_dyn', 'x_dyn'], ['y_out'], {}, 'fdiv1'));
    gSub.nodes.push(new Node('CompletelyUnknownOp', ['x_dyn'], ['y_out'], {}, 'unk1'));

    compileGraphToTFLite(gSub, new TFLiteExporter(), true, 'none');
    process.env.TFLITE_STRIP_CUSTOM_OPS = '1';
    try {
      compileGraphToTFLite(gSub, new TFLiteExporter(), true, 'none');
    } finally {
      delete process.env.TFLITE_STRIP_CUSTOM_OPS;
    }

    // layout.ts exhaustive edge cases
    const gLay = new Graph('lay');
    (gLay as unknown as { metadata: { producer_name: string } }).metadata = {
      producer_name: 'onnx9000.keras',
    };
    gLay.valueInfo.push(new ValueInfo('hidden_state', [1, 4], 'float32'));
    gLay.inputs.push(new ValueInfo('in_1d', [1, 2, 8], 'float32'));
    gLay.outputs.push(new ValueInfo('id_graph_out', [1, 2, 8], 'float32'));
    gLay.tensors.w_1d = new Tensor('w_1d', [2, 2, 3], 'float32', true, false, new Float32Array(12));
    gLay.nodes.push(new Node('Identity', ['in_1d'], ['id_graph_out'], {}, 'id_out_node'));
    gLay.nodes.push(new Node('Conv', [], ['no_in_out'], {}, 'conv_no_in'));
    gLay.nodes.push(new Node('Conv', ['in_1d'], [], {}, 'conv_1d_no_out'));
    gLay.nodes.push(
      new Node(
        'Conv',
        ['in_1d', 'w_1d'],
        ['out_1d'],
        {
          kernel_shape: new Attribute('kernel_shape', 'INTS', [3]),
          strides: new Attribute('strides', 'INTS', [1]),
          dilations: new Attribute('dilations', 'INTS', [1]),
          pads: new Attribute('pads', 'INTS', [1, 1]),
        },
        'conv_1d',
      ),
    );
    gLay.nodes.push(new Node('Einsum', ['in_1d', 'in_1d'], ['ein_no_eq'], {}, 'ein0'));
    gLay.nodes.push(
      new Node(
        'Einsum',
        ['in_1d', 'in_1d'],
        ['ein_out'],
        { equation: new Attribute('equation', 'STRING', 'ij,jk->ik') },
        'ein1',
      ),
    );
    // Conv + BN edge cases:
    // 1) Conv with [] outputs
    // 2) Conv output with 2 consumers
    // 3) Conv followed by BN with < 5 inputs
    // 4) Conv with 1 input followed by BN
    // 5) Conv with mismatched output channels
    // 6) Conv with bias tensor that has null data, and Conv with bias tensor that has valid data
    // 7) BN placed BEFORE Conv in graph.nodes
    gLay.tensors.w_conv = new Tensor(
      'w_conv',
      [2, 2, 1, 1],
      'float32',
      true,
      false,
      new Float32Array([1, 1, 1, 1]),
    );
    gLay.tensors.w_mismatch = new Tensor(
      'w_mismatch',
      [1, 2, 1, 1],
      'float32',
      true,
      false,
      new Float32Array([1, 1]),
    );
    gLay.tensors.b_null = new Tensor('b_null', [2], 'float32', true, false, null);
    gLay.tensors.b_valid = new Tensor(
      'b_valid',
      [2],
      'float32',
      true,
      false,
      new Float32Array([0.5, 0.5]),
    );
    gLay.tensors.bn_s = new Tensor('bn_s', [2], 'float32', true, false, new Float32Array([1, 1]));
    gLay.tensors.bn_b = new Tensor('bn_b', [2], 'float32', true, false, new Float32Array([0, 0]));
    gLay.tensors.bn_m = new Tensor('bn_m', [2], 'float32', true, false, new Float32Array([0, 0]));
    gLay.tensors.bn_v = new Tensor('bn_v', [2], 'float32', true, false, new Float32Array([1, 1]));
    // BN before Conv (with valid bias on Conv!)
    gLay.nodes.push(
      new Node(
        'BatchNormalization',
        ['conv_before_out', 'bn_s', 'bn_b', 'bn_m', 'bn_v'],
        ['bn_before_out'],
        {},
        'bn_before',
      ),
    );
    gLay.nodes.push(
      new Node('Conv', ['x4d', 'w_conv', 'b_valid'], ['conv_before_out'], {}, 'conv_after_bn'),
    );
    gLay.nodes.push(new Node('Conv', ['x4d', 'w_conv'], ['conv_nobias_out'], {}, 'conv_nobias'));
    gLay.nodes.push(
      new Node(
        'BatchNormalization',
        ['conv_nobias_out', 'bn_s', 'bn_b', 'bn_m', 'bn_v'],
        ['bn_nobias_out'],
        {},
        'bn_nobias',
      ),
    );
    // Conv with 2 consumers
    gLay.nodes.push(new Node('Conv', ['x4d', 'w_conv'], ['conv_2cons'], {}, 'c_2cons'));
    gLay.nodes.push(
      new Node(
        'BatchNormalization',
        ['conv_2cons', 'bn_s', 'bn_b', 'bn_m', 'bn_v'],
        ['bn_2c'],
        {},
        'bn_2c',
      ),
    );
    gLay.nodes.push(new Node('Relu', ['conv_2cons'], ['relu_2c'], {}, 'relu_2c'));
    // Conv followed by BN with < 5 inputs
    gLay.nodes.push(new Node('Conv', ['x4d', 'w_conv'], ['conv_short_bn'], {}, 'c_short_bn'));
    gLay.nodes.push(
      new Node('BatchNormalization', ['conv_short_bn'], ['bn_short_out'], {}, 'bn_s'),
    );
    // Conv with 1 input followed by BN
    gLay.nodes.push(new Node('Conv', ['x4d'], ['conv_1in_out'], {}, 'c_1in'));
    gLay.nodes.push(
      new Node(
        'BatchNormalization',
        ['conv_1in_out', 'bn_s', 'bn_b', 'bn_m', 'bn_v'],
        ['bn_1in_out'],
        {},
        'bn_1in',
      ),
    );
    // Conv with mismatched channels
    gLay.nodes.push(new Node('Conv', ['x4d', 'w_mismatch'], ['conv_mis_out'], {}, 'c_mis'));
    gLay.nodes.push(
      new Node(
        'BatchNormalization',
        ['conv_mis_out', 'bn_s', 'bn_b', 'bn_m', 'bn_v'],
        ['bn_mis_out'],
        {},
        'bn_mis',
      ),
    );
    // Conv with null bias data
    gLay.nodes.push(new Node('Conv', ['x4d', 'w_conv', 'b_null'], ['conv_bnull'], {}, 'c_bnull'));
    gLay.nodes.push(
      new Node(
        'BatchNormalization',
        ['conv_bnull', 'bn_s', 'bn_b', 'bn_m', 'bn_v'],
        ['bn_bnull_out'],
        {},
        'bn_bnull',
      ),
    );
    // Standalone BN to decompose
    gLay.nodes.push(
      new Node(
        'BatchNormalization',
        ['x_standalone', 'bn_s', 'bn_b', 'bn_m', 'bn_v'],
        ['bn_decomp_out'],
        {},
        'bn_decomp',
      ),
    );
    // Div by zero
    gLay.tensors.zero_div = new Tensor(
      'zero_div',
      [1],
      'float32',
      true,
      false,
      new Float32Array([0]),
    );
    gLay.nodes.push(new Node('Div', ['x_standalone', 'zero_div'], ['div_out'], {}, 'div_zero'));
    // LSTM, Resize, ConvTranspose, Gemm (1 input, transB=1, and int32)
    gLay.nodes.push(new Node('LSTM', ['x_standalone'], ['lstm_out'], {}, 'lstm1'));
    gLay.tensors.sizes_i64 = new Tensor(
      'sizes_i64',
      [4],
      'int64',
      true,
      false,
      new BigInt64Array(4),
    );
    gLay.nodes.push(
      new Node('Resize', ['x_standalone', '', '', 'sizes_i64'], ['res_i64_out'], {}, 'res_i64'),
    );
    gLay.inputs.push(new ValueInfo('x_dyn_res', [-1, 3, 4, 4], 'float32'));
    gLay.tensors.scales_f32 = new Tensor(
      'scales_f32',
      [4],
      'float32',
      true,
      false,
      new Float32Array([1, 1, 2, 2]),
    );
    gLay.nodes.push(
      new Node('Resize', ['x_dyn_res', '', 'scales_f32'], ['res_sc_out'], {}, 'res_scales'),
    );
    gLay.nodes.push(new Node('Resize', ['', '', 'scales_f32'], ['res_no_in_out'], {}, 'res_no_in'));
    gLay.tensors.w_ct = new Tensor(
      'w_ct',
      [2, 2, 1, 1],
      'float32',
      true,
      false,
      new Uint8Array(16),
    );
    gLay.nodes.push(
      new Node(
        'ConvTranspose',
        ['x_dyn_res', 'w_ct'],
        ['ct_out'],
        { output_padding: new Attribute('output_padding', 'INTS', [1, 1]) },
        'ct1',
      ),
    );
    gLay.tensors.w_gemm_i32 = new Tensor(
      'w_gemm_i32',
      [2, 2],
      'int32',
      true,
      false,
      new Int32Array([1, 2, 3, 4]),
    );
    gLay.nodes.push(new Node('Gemm', ['x_standalone'], ['gemm_1in_out'], {}, 'g_1in'));
    gLay.nodes.push(
      new Node(
        'Gemm',
        ['x_standalone', 'w_gemm_i32'],
        ['gemm_tb1_out'],
        { transB: new Attribute('transB', 'INT', 1) },
        'g_tb1',
      ),
    );
    gLay.nodes.push(
      new Node('Gemm', ['x_standalone', 'w_gemm_i32'], ['gemm_i32_out'], {}, 'g_i32'),
    );
    // Add placed BEFORE MatMul, and Relu placed BEFORE Conv
    gLay.nodes.push(
      new Node('Add', ['mm_before_y', 'mm_bias'], ['mm_before_add_y'], {}, 'add_bef'),
    );
    gLay.nodes.push(new Node('MatMul', ['mm_a', 'mm_b'], ['mm_before_y'], {}, 'mm_aft'));
    gLay.nodes.push(new Node('Relu', ['conv_before_relu_y'], ['relu_bef_y'], {}, 'relu_bef'));
    gLay.nodes.push(new Node('Conv', ['x4d', 'w_conv'], ['conv_before_relu_y'], {}, 'conv_aft'));
    // Negative axis and axes rewriting (with known rank in inputs)
    gLay.nodes.push(
      new Node(
        'ReduceMean',
        ['in_1d'],
        ['rm_known'],
        {
          axis: new Attribute('axis', 'INT', -1),
          axes: new Attribute('axes', 'INTS', [-1]),
        },
        'rm_known_node',
      ),
    );
    gLay.nodes.push(
      new Node(
        'ReduceMean',
        ['unknown_rank_in'],
        ['rm1'],
        {
          axis: new Attribute('axis', 'INT', -1),
          axes: new Attribute('axes', 'INTS', [-1, 5]),
        },
        'rm_unk',
      ),
    );
    // injectTransposes with rank 2 (<3) and rank 5 (NCDHW)
    gLay.inputs.push(new ValueInfo('in_r2', [2, 2], 'float32'));
    gLay.inputs.push(new ValueInfo('in_r5', [1, 2, 3, 4, 5], 'float32'));
    gLay.nodes.push(new Node('MaxPool', ['in_r2'], ['out_r2'], {}, 'mp_r2'));
    gLay.nodes.push(new Node('MaxPool', ['in_r5'], ['out_r5'], {}, 'mp_r5'));

    const opt = new LayoutOptimizer(gLay, false);
    opt.optimize();

    // Direct calls on private methods for states after/between passes
    const optAny = opt as unknown as {
      injectTransposes: () => void;
      pushDownTransposes: () => void;
      cancelTransposes: () => void;
      foldConstants: () => void;
      transposeTensorData: (t: Tensor, perm: number[]) => void;
    };

    const gDirect = new Graph('direct');
    gDirect.inputs.push(new ValueInfo('in_r3', [1, 2, 8], 'float32'));
    gDirect.outputs.push(new ValueInfo('cancel_final_out', [1, 2, 3, 4], 'float32'));
    gDirect.nodes.push(new Node('MaxPool', ['in_r3'], ['out_r3'], {}, 'mp_r3'));
    gDirect.nodes.push(new Node('MaxPool', ['in_r3'], [], {}, 'mp_no_out'));
    const optDirect = new LayoutOptimizer(gDirect, false) as unknown as typeof optAny;
    optDirect.injectTransposes();

    const gPush = new Graph('push');
    gPush.nodes.push(
      new Node(
        'Transpose',
        ['p0'],
        ['t_other_perm'],
        { perm: new Attribute('perm', 'INTS', [0, 1, 2, 3]) },
        'tr_other',
      ),
    );
    gPush.nodes.push(new Node('Relu', ['t_other_perm'], ['r_other'], {}, 'relu_other'));
    gPush.nodes.push(
      new Node(
        'Transpose',
        ['p1'],
        ['t_nhwc'],
        { perm: new Attribute('perm', 'INTS', [0, 2, 3, 1]) },
        'tr_nhwc',
      ),
    );
    gPush.nodes.push(
      new Node(
        'Transpose',
        ['p2'],
        ['t_nchw'],
        { perm: new Attribute('perm', 'INTS', [0, 3, 1, 2]) },
        'tr_nchw',
      ),
    );
    // Add with mismatched transpose inputs
    gPush.nodes.push(new Node('Add', ['t_nhwc', 't_nchw'], ['add_mis'], {}, 'add_mis'));
    // ReduceMean with negative axis & axes after 0,2,3,1 and after 0,3,1,2, and Expand, and [''] output
    gPush.nodes.push(new Node('Expand', ['t_nhwc'], ['exp_nhwc'], {}, 'exp_nhwc'));
    gPush.nodes.push(
      new Node(
        'ReduceMean',
        ['exp_nhwc'],
        ['rm_nhwc'],
        {
          axis: new Attribute('axis', 'INT', -1),
          axes: new Attribute('axes', 'INTS', [-1, 5]),
        },
        'rm_nhwc',
      ),
    );
    gPush.nodes.push(
      new Node(
        'ReduceMean',
        ['t_nchw'],
        ['rm_nchw'],
        {
          axis: new Attribute('axis', 'INT', -1),
          axes: new Attribute('axes', 'INTS', [-1, 5]),
        },
        'rm_nchw',
      ),
    );
    // Relu with empty output string
    gPush.nodes.push(new Node('Relu', ['rm_nchw'], [''], {}, 'relu_empty_out'));
    const optPush = new LayoutOptimizer(gPush, false) as unknown as typeof optAny;
    optPush.pushDownTransposes();

    const gCancel = new Graph('cancel');
    gCancel.outputs.push(new ValueInfo('t_cancel_out', [1, 2, 3, 4], 'float32'));
    gCancel.nodes.push(new Node('Transpose', ['x0'], ['t_noperm'], {}, 'tr_noperm1'));
    gCancel.nodes.push(new Node('Transpose', ['t_noperm'], ['t_noperm2'], {}, 'tr_noperm2'));
    gCancel.nodes.push(
      new Node(
        'Transpose',
        ['x1'],
        ['t_mid'],
        { perm: new Attribute('perm', 'INTS', [0, 2, 3, 1]) },
        'tr_c1',
      ),
    );
    gCancel.nodes.push(
      new Node(
        'Transpose',
        ['t_mid'],
        ['t_cancel_out'],
        { perm: new Attribute('perm', 'INTS', [0, 3, 1, 2]) },
        'tr_c2',
      ),
    );
    const optCancel = new LayoutOptimizer(gCancel, false) as unknown as typeof optAny;
    optCancel.cancelTransposes();
    expect(gCancel.outputs[0]?.name).toBe('x1');

    const gFold = new Graph('fold');
    gFold.nodes.push(new Node('BatchNormalization', ['x'], ['y'], {}, 'bn_in_fold'));
    const optFold = new LayoutOptimizer(gFold, false) as unknown as typeof optAny;
    optFold.foldConstants();
    optFold.transposeTensorData(new Tensor('null_t', [2, 2], 'float32', true, false, null), [1, 0]);

    warnSpy.mockRestore();
    logSpy.mockRestore();
  });
});
