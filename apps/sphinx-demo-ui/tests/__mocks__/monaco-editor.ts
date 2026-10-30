import { vi } from 'vitest';

export const editor = {
  create: vi.fn(() => ({
    setValue: vi.fn(),
    getValue: vi.fn().mockReturnValue('mock code'),
    onDidChangeModelContent: vi.fn().mockReturnValue({ dispose: vi.fn() }),
    layout: vi.fn(),
    setModel: vi.fn(),
    dispose: vi.fn(),
  })),
  createModel: vi.fn((content: string) => ({
    getValue: vi.fn().mockReturnValue(content),
    setValue: vi.fn(),
    dispose: vi.fn(),
  })),
  setTheme: vi.fn(),
};

export const Uri = {
  parse: vi.fn((str: string) => ({ path: str })),
};

export default class MockWorker {}
