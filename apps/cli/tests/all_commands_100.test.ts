import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import * as convertersMod from '@onnx9000/converters';
import * as coreMod from '@onnx9000/core';
import * as serveMod from '@onnx9000/serve';
import * as transformersMod from '@onnx9000/transformers';
import { describe, expect, it, vi } from 'vitest';
import { handleAppleCommand } from '../src/commands/apple.js';
import { handleArena } from '../src/commands/arena.js';
import { handleConvertCommand } from '../src/commands/convert.js';
import { handleCudaCommand } from '../src/commands/cuda.js';
import { handleDiffusersCommand } from '../src/commands/diffusers.js';
import { handleExportCommand } from '../src/commands/export.js';
import { handleInspectCommand } from '../src/commands/inspect.js';
import { handleJaxCommand } from '../src/commands/jax.js';
import { handleJsonExtractCommand } from '../src/commands/json-extract.js';
import { handleKeras2ONNX } from '../src/commands/keras2onnx.js';
import { handleMlirCommand } from '../src/commands/mlir.js';
import { handleMmdnnCommand } from '../src/commands/mmdnn.js';
import { handleMobileMemoryCommand } from '../src/commands/mobile-memory.js';
import { handleNewModelArchCommand } from '../src/commands/new-model-arch.js';
import { handleOnnxCheckerCommand } from '../src/commands/onnx-checker.js';
import { handleOnnx2cCommand } from '../src/commands/onnx2c.js';
import { handleOptimumCommand } from '../src/commands/optimum.js';
import { handlePaddle2ONNX } from '../src/commands/paddle2onnx.js';
import { handleProgressiveLoadingCommand } from '../src/commands/progressive-loading.js';
import { handlePytorchCodegenCommand } from '../src/commands/pytorch-codegen.js';
import { handleRocmCommand } from '../src/commands/rocm.js';
import { handleScriptCommand } from '../src/commands/script.js';
import { handleSKL2ONNX } from '../src/commands/skl2onnx.js';
import { handleSparseCommand } from '../src/commands/sparse.js';
import { handleTensorRTCommand } from '../src/commands/tensorrt.js';
import { handleTfliteCommand } from '../src/commands/tflite.js';
import { handleTransformersCommand } from '../src/commands/transformers.js';
import { handleTvmCommand } from '../src/commands/tvm.js';
import { handleWasmCommand } from '../src/commands/wasm.js';
import { handleWebgpuCommand } from '../src/commands/webgpu.js';
import { handleWebnnPolyfillCommand } from '../src/commands/webnn-polyfill.js';
import { handleZeroDepClassifierCommand } from '../src/commands/zero-dep-classifier.js';
import { handleZooCommand } from '../src/commands/zoo.js';
import { __test_main__ } from '../src/index.js';

vi.mock('node:fs', async (importOriginal) => {
  const actual = await importOriginal<typeof import('node:fs')>();
  return {
    ...actual,
    statSync: vi.fn((...args: Parameters<typeof actual.statSync>) => actual.statSync(...args)),
    existsSync: vi.fn((...args: Parameters<typeof actual.existsSync>) =>
      actual.existsSync(...args),
    ),
  };
});

vi.mock('@onnx9000/core', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@onnx9000/core')>();
  return {
    ...actual,
    load: vi.fn(),
  };
});

describe('apps/cli exhaustive 100% coverage', () => {
  it('covers simple CLI command handlers', async () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const exitSpy = vi.spyOn(process, 'exit').mockImplementation((() => {}) as never);

    try {
      for (const fn of [
        handleAppleCommand,
        handleCudaCommand,
        handleJaxCommand,
        handleMlirCommand,
        handleMmdnnCommand,
        handleMobileMemoryCommand,
        handleNewModelArchCommand,
        handleOnnxCheckerCommand,
        handleProgressiveLoadingCommand,
        handleRocmCommand,
        handleTensorRTCommand,
        handleTvmCommand,
        handleWasmCommand,
        handleWebgpuCommand,
        handleZeroDepClassifierCommand,
      ]) {
        fn([]);
        fn(['-h']);
        fn(['--help']);
        fn(['model.onnx']);
      }

      handleOnnx2cCommand([]);
      handleOnnx2cCommand(['model.onnx']);
      handleOnnx2cCommand(['model.onnx', '-o', 'custom.c']);

      handleScriptCommand([]);
      handleScriptCommand(['script.py']);
      handleScriptCommand(['script.py', '-o', 'out.onnx']);

      handleWebnnPolyfillCommand(['-h']);
      handleWebnnPolyfillCommand(['--help']);
      handleWebnnPolyfillCommand([]);

      for (const fn of [handleArena, handleKeras2ONNX, handlePaddle2ONNX, handleSKL2ONNX]) {
        fn([]);
        fn(['model.onnx']);
      }

      handleDiffusersCommand([]);
      handleDiffusersCommand(['model', '--prompt', 'a cat']);
      const tmpOut = path.join(os.tmpdir(), `diff_${Date.now()}.png`);
      handleDiffusersCommand(['model', '--prompt', 'a cat', '-o', tmpOut]);
      handleDiffusersCommand(['model', '--prompt', 'a cat', '--output']);
      if (fs.existsSync(tmpOut)) fs.unlinkSync(tmpOut);

      await handleTfliteCommand(['-h']);
      handleExportCommand(['model.onnx']); // missing --format
      handleOptimumCommand(['export', 'm1']); // without --task
      handleOptimumCommand(['optimize', 'm1']); // without --level
      handleOptimumCommand(['quantize', 'm1']); // without --quantize
      handleSparseCommand(['prune', 'm1.onnx', '--recipe', 'r.yaml']);
      handleZooCommand(['pull']); // missing modelId
    } finally {
      logSpy.mockRestore();
      errSpy.mockRestore();
      exitSpy.mockRestore();
    }
  });

  it('covers inspect, convert, transformers, gguf, json-extract, and pytorch-codegen', async () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const exitSpy = vi.spyOn(process, 'exit').mockImplementation((() => {
      throw new Error('exit');
    }) as never);

    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'onnx9000-cli-all-'));
    try {
      // inspect.ts
      await expect(handleInspectCommand([])).rejects.toThrow('exit');
      await expect(handleInspectCommand([path.join(tmpDir, 'missing.keras')])).rejects.toThrow(
        'exit',
      );
      const kerasFile = path.join(tmpDir, 'm.keras');
      const otherFile = path.join(tmpDir, 'm.txt');
      fs.writeFileSync(kerasFile, 'dummy');
      fs.writeFileSync(otherFile, 'dummy');
      await handleInspectCommand([kerasFile]);
      await handleInspectCommand([otherFile]);

      // convert.ts
      exitSpy.mockImplementation((() => {}) as never);
      await handleConvertCommand([]);
      await handleConvertCommand(['--src']);
      await handleConvertCommand(['--unknown-flag', '--dst']);
      await handleConvertCommand(['--src', 'caffe', '--dst', 'onnx']);
      await handleConvertCommand(['--from', '--to']);

      const subDir = path.join(tmpDir, 'batch');
      fs.mkdirSync(subDir);
      fs.mkdirSync(path.join(subDir, 'nested_dir'));
      const f1 = path.join(subDir, 'a.prototxt');
      const f2 = path.join(subDir, 'a.caffemodel');
      fs.writeFileSync(f1, 'p');
      fs.writeFileSync(f2, 'c');

      const convSpy = vi
        .spyOn(convertersMod.mmdnn, 'convert')
        .mockImplementationOnce(async (_s, _d, blobs: File[]) => {
          const b0 = blobs[0] as File;
          b0.stream().destroy();
          await b0.arrayBuffer();
          b0.slice(0, 1);
          b0.slice();
          return 'converted_str';
        })
        .mockResolvedValueOnce({ graph: true } as never)
        .mockRejectedValueOnce(new Error('conv fail'));

      try {
        await handleConvertCommand(['--from', 'caffe', '--to', 'onnx', subDir]);
        await handleConvertCommand(['--src', 'caffe', '--dst', 'onnx', f1]);
        await handleConvertCommand(['--src', 'caffe', '--dst', 'onnx', f1]);
      } finally {
        convSpy.mockRestore();
        if (fs.existsSync('a_converted.out')) fs.unlinkSync('a_converted.out');
      }

      // transformers.ts
      await handleTransformersCommand([]);
      await handleTransformersCommand(['--help']);
      await handleTransformersCommand(['-h']);
      const pipeSpy = vi
        .spyOn(transformersMod, 'pipeline')
        .mockResolvedValueOnce((async (x: string) => [{ label: x }]) as never)
        .mockRejectedValueOnce(new Error('pipe fail'))
        .mockRejectedValueOnce('raw pipe fail');
      await handleTransformersCommand(['text-classification']);
      await handleTransformersCommand(['text-classification', 'hello', 'world']);
      await handleTransformersCommand(['text-classification', 'fail']);
      pipeSpy.mockRestore();

      // json-extract.ts & pytorch-codegen.ts with --output
      vi.mocked(coreMod.load).mockResolvedValue({
        name: 'g',
        tensors: {},
        inputs: [],
        outputs: [],
        nodes: [],
        initializers: [],
        valueInfo: [],
      } as never);
      await handleJsonExtractCommand([f1, '--output', path.join(tmpDir, 'j.json')]);
      await handlePytorchCodegenCommand([f1, '--output', path.join(tmpDir, 'p.py')]);
    } finally {
      fs.rmSync(tmpDir, { recursive: true, force: true });
      logSpy.mockRestore();
      errSpy.mockRestore();
      exitSpy.mockRestore();
    }
  });

  it('covers index.ts __test_main__ dispatch for every command', async () => {
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const errSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const exitSpy = vi.spyOn(process, 'exit').mockImplementation((() => {}) as never);
    const serveSpy = vi.spyOn(serveMod, 'runCli').mockImplementation(() => {});

    const origArgv = process.argv;
    try {
      const commands = [
        ['convert'],
        ['inspect', 'nonexistent.keras'],
        ['json-extract', '-h'],
        ['pytorch-codegen', '-h'],
        ['whisper-llm', '-h'],
        ['llama-web', '-h'],
        ['tfjs-shim', '-h'],
        ['onnx2gguf', '--dry-run', 'm.onnx'],
        ['gguf2onnx'],
        ['tvm', '-h'],
        ['tensorrt', '-h'],
        ['diffusers', '-h'],
        ['serve'],
        ['array'],
        ['iree', '-h'],
        ['triton', '-h'],
        ['openvino'],
        ['transformers', '-h'],
        ['coreml', '-h'],
        ['autograd', '-h'],
        ['zoo', '-h'],
        ['hummingbird', '-h'],
        ['sparse', '-h'],
        ['optimum', '-h'],
        ['sphinx-demo-ui', '-h'],
        ['export', '-h'],
        ['onnx2tf', '-h'],
        ['optimize', '-h'],
        ['simplify', '-h'],
        ['agent', '-h'],
        ['apple', '-h'],
        ['onnx2c', '-h'],
        ['cuda', '-h'],
        ['jax', '-h'],
        ['mmdnn', '-h'],
        ['rocm', '-h'],
        ['wasm', '-h'],
        ['webgpu', '-h'],
        ['webnn-polyfill', '-h'],
        ['onnx-checker', '-h'],
        ['script', '-h'],
        ['mlir', '-h'],
        ['mobile-memory', '-h'],
        ['progressive-loading', '-h'],
        ['tflite', '-h'],
        ['new-model-arch', '-h'],
        ['zero-dep-classifier', '-h'],
        ['paddle2onnx', 'm'],
        ['keras2onnx', 'm'],
        ['skl2onnx', 'm'],
        ['arena', 'm'],
        ['unknown-command'],
      ];

      for (const cmdArgs of commands) {
        process.argv = ['node', 'onnx9000', ...cmdArgs];
        try {
          await __test_main__();
        } catch {
          // ignore mocked exits/errors
        }
      }

      vi.mocked(fs.existsSync).mockReturnValueOnce(false);
      const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue('/');
      process.argv = ['node', 'onnx9000', 'edit'];
      await __test_main__();
      cwdSpy.mockRestore();
      expect(serveSpy).toHaveBeenCalled();
    } finally {
      process.argv = origArgv;
      serveSpy.mockRestore();
      logSpy.mockRestore();
      errSpy.mockRestore();
      exitSpy.mockRestore();
    }
  });
});
