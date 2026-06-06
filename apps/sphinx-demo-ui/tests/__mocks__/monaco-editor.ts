import { vi } from 'vitest';

export const editor = {
  create: vi.fn(() => ({
    setValue: vi.fn(),
    getValue: vi.fn().mockReturnValue('mock code'),
    onDidChangeModelContent: vi.fn().mockReturnValue({ dispose: vi.fn() }),
    dispose: vi.fn(),
  })),
  setTheme: vi.fn(),
};
