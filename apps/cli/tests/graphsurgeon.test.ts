import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { handleOptimizeCommand } from '../src/commands/optimize.js';
import { handleSimplifyCommand } from '../src/commands/simplify.js';

describe('Graphsurgeon commands', () => {
  let consoleLogSpy: any;
  let processExitSpy: any;

  beforeEach(() => {
    consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    processExitSpy = vi.spyOn(process, 'exit').mockImplementation((() => {}) as any);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('handleOptimizeCommand', () => {
    it('should print help and exit if no args or -h', async () => {
      await handleOptimizeCommand([]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Usage: onnx9000 optimize'),
      );
      expect(processExitSpy).toHaveBeenCalledWith(0);

      await handleOptimizeCommand(['-h']);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Usage: onnx9000 optimize'),
      );
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it('should run optimize command with default output', async () => {
      await handleOptimizeCommand(['model.onnx']);
      expect(consoleLogSpy).toHaveBeenCalledWith('Loading ONNX model model.onnx...');
      expect(consoleLogSpy).toHaveBeenCalledWith('Running optimization passes: default');
      expect(consoleLogSpy).toHaveBeenCalledWith('Saving optimized model to model_opt.onnx...');
      expect(consoleLogSpy).toHaveBeenCalledWith('Graph optimization complete.');
    });

    it('should run optimize command with custom output and passes', async () => {
      await handleOptimizeCommand([
        'model.onnx',
        '--passes',
        'fuse_bn_into_conv',
        '-o',
        'out.onnx',
      ]);
      expect(consoleLogSpy).toHaveBeenCalledWith('Loading ONNX model model.onnx...');
      expect(consoleLogSpy).toHaveBeenCalledWith('Running optimization passes: fuse_bn_into_conv');
      expect(consoleLogSpy).toHaveBeenCalledWith('Saving optimized model to out.onnx...');
    });
  });

  describe('handleSimplifyCommand', () => {
    it('should print help and exit if no args or -h', async () => {
      await handleSimplifyCommand([]);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Usage: onnx9000 simplify'),
      );
      expect(processExitSpy).toHaveBeenCalledWith(0);

      await handleSimplifyCommand(['-h']);
      expect(consoleLogSpy).toHaveBeenCalledWith(
        expect.stringContaining('Usage: onnx9000 simplify'),
      );
      expect(processExitSpy).toHaveBeenCalledWith(0);
    });

    it('should run simplify command with default output', async () => {
      await handleSimplifyCommand(['model.onnx']);
      expect(consoleLogSpy).toHaveBeenCalledWith('Loading ONNX model model.onnx...');
      expect(consoleLogSpy).toHaveBeenCalledWith('Simplifying graph...');
      expect(consoleLogSpy).toHaveBeenCalledWith('Saving simplified model to model_sim.onnx...');
      expect(consoleLogSpy).toHaveBeenCalledWith('Graph simplification complete.');
    });

    it('should run simplify command with custom output', async () => {
      await handleSimplifyCommand(['model.onnx', '-o', 'out.onnx']);
      expect(consoleLogSpy).toHaveBeenCalledWith('Loading ONNX model model.onnx...');
      expect(consoleLogSpy).toHaveBeenCalledWith('Simplifying graph...');
      expect(consoleLogSpy).toHaveBeenCalledWith('Saving simplified model to out.onnx...');
    });
  });
});
