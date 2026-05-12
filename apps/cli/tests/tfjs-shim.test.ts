import { describe, it, expect, vi } from 'vitest';
import { handleTfjsShimCommand } from '../src/commands/tfjs-shim.js';

describe('handleTfjsShimCommand', () => {
  it('should print help and exit if -h', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const exitSpy = vi.spyOn(process, 'exit').mockImplementation((() => {}) as any);

    await handleTfjsShimCommand(['-h']);
    expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Usage:'));
    expect(exitSpy).toHaveBeenCalledWith(0);

    consoleSpy.mockRestore();
    exitSpy.mockRestore();
  });

  it('should print verification', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    await handleTfjsShimCommand([]);
    expect(consoleSpy).toHaveBeenCalledWith('TFJS Shim environment verified.');
    consoleSpy.mockRestore();
  });
});

it('should handle zero args by continuing to verification', async () => {
  const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
  await handleTfjsShimCommand([]);
  expect(consoleSpy).toHaveBeenCalledWith('TFJS Shim environment verified.');
  consoleSpy.mockRestore();
});

it('should handle some non-help args by bypassing help', async () => {
  const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
  await handleTfjsShimCommand(['somethingelse']);
  expect(consoleSpy).toHaveBeenCalledWith('TFJS Shim environment verified.');
  consoleSpy.mockRestore();
});
