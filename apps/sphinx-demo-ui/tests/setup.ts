/* eslint-disable */
// @ts-nocheck
import { vi } from 'vitest';

// Mock ResizeObserver for JSDOM
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
} as object;

// Mock Monaco Editor
vi.mock('monaco-editor', () => ({
  editor: {
    create: vi.fn(() => ({
      getValue: vi.fn().mockReturnValue('mocked content'),
      setValue: vi.fn(),
      layout: vi.fn(),
      dispose: vi.fn(),
      onDidChangeModelContent: vi.fn(() => ({ dispose: vi.fn() })),
      setModel: vi.fn()
    })),
    createModel: vi.fn((content, _lang, _uri) => ({
      getValue: vi.fn().mockReturnValue(content),
      setValue: vi.fn(),
      dispose: vi.fn()
    })),
    setTheme: vi.fn()
  },
  Uri: {
    parse: vi.fn((str) => ({ path: str }))
  }
}));

// Mock window location to bypass JSDOM navigation issues
Object.defineProperty(window, 'location', {
  value: {
    href: 'http://localhost/',
    reload: vi.fn()
  },
  writable: true
});

const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] || null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = value.toString();
    }),
    removeItem: vi.fn((key: string) => {
      delete store[key];
    }),
    clear: vi.fn(() => {
      store = {};
    })
  };
})();

Object.defineProperty(globalThis, 'localStorage', {
  value: localStorageMock,
  writable: true,
  configurable: true
});
if (typeof window !== 'undefined') {
  Object.defineProperty(window, 'localStorage', {
    value: localStorageMock,
    writable: true,
    configurable: true
  });
}
