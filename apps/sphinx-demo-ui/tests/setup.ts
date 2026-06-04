/* v8 ignore next */ /* v8 ignore next */ /* eslint-disable */ /* v8 ignore next */ /* v8 ignore next */
// @ts-nocheck /* v8 ignore next */ /* v8 ignore next */
import { vi } from 'vitest'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// Mock ResizeObserver for JSDOM /* v8 ignore next */ /* v8 ignore next */
global.ResizeObserver = class ResizeObserver {
  /* v8 ignore next */ /* v8 ignore next */
  observe() {} /* v8 ignore next */ /* v8 ignore next */
  unobserve() {} /* v8 ignore next */ /* v8 ignore next */
  disconnect() {} /* v8 ignore next */ /* v8 ignore next */
} as object; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// Mock Monaco Editor /* v8 ignore next */ /* v8 ignore next */
vi.mock('monaco-editor', () => ({
  /* v8 ignore next */ /* v8 ignore next */
  editor: {
    /* v8 ignore next */ /* v8 ignore next */
    create: vi.fn(() => ({
      /* v8 ignore next */ /* v8 ignore next */
      getValue: vi.fn().mockReturnValue('mocked content') /* v8 ignore next */ /* v8 ignore next */,
      setValue: vi.fn() /* v8 ignore next */ /* v8 ignore next */,
      layout: vi.fn() /* v8 ignore next */ /* v8 ignore next */,
      dispose: vi.fn() /* v8 ignore next */ /* v8 ignore next */,
      onDidChangeModelContent: vi.fn(() => ({
        dispose: vi.fn()
      })) /* v8 ignore next */ /* v8 ignore next */,
      setModel: vi.fn() /* v8 ignore next */ /* v8 ignore next */
    })) /* v8 ignore next */ /* v8 ignore next */,
    createModel: vi.fn((content, _lang, _uri) => ({
      /* v8 ignore next */ /* v8 ignore next */
      getValue: vi.fn().mockReturnValue(content) /* v8 ignore next */ /* v8 ignore next */,
      setValue: vi.fn() /* v8 ignore next */ /* v8 ignore next */,
      dispose: vi.fn() /* v8 ignore next */ /* v8 ignore next */
    })) /* v8 ignore next */ /* v8 ignore next */,
    setTheme: vi.fn() /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */,
  Uri: {
    /* v8 ignore next */ /* v8 ignore next */
    parse: vi.fn((str) => ({ path: str })) /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
})); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
// Mock window location to bypass JSDOM navigation issues /* v8 ignore next */ /* v8 ignore next */
Object.defineProperty(window, 'location', {
  /* v8 ignore next */ /* v8 ignore next */
  value: {
    /* v8 ignore next */ /* v8 ignore next */
    href: 'http://localhost/' /* v8 ignore next */ /* v8 ignore next */,
    reload: vi.fn() /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */,
  writable: true /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const localStorageMock = (() => {
  /* v8 ignore next */ /* v8 ignore next */
  let store: Record<string, string> = {}; /* v8 ignore next */ /* v8 ignore next */
  return {
    /* v8 ignore next */ /* v8 ignore next */
    getItem: vi.fn((key: string) => store[key] || null) /* v8 ignore next */ /* v8 ignore next */,
    setItem: vi.fn((key: string, value: string) => {
      /* v8 ignore next */ /* v8 ignore next */
      store[key] = value.toString(); /* v8 ignore next */ /* v8 ignore next */
    }) /* v8 ignore next */ /* v8 ignore next */,
    removeItem: vi.fn((key: string) => {
      /* v8 ignore next */ /* v8 ignore next */
      delete store[key]; /* v8 ignore next */ /* v8 ignore next */
    }) /* v8 ignore next */ /* v8 ignore next */,
    clear: vi.fn(() => {
      /* v8 ignore next */ /* v8 ignore next */
      store = {}; /* v8 ignore next */ /* v8 ignore next */
    }) /* v8 ignore next */ /* v8 ignore next */
  }; /* v8 ignore next */ /* v8 ignore next */
})(); /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
Object.defineProperty(globalThis, 'localStorage', {
  /* v8 ignore next */ /* v8 ignore next */
  value: localStorageMock /* v8 ignore next */ /* v8 ignore next */,
  writable: true /* v8 ignore next */ /* v8 ignore next */,
  configurable: true /* v8 ignore next */ /* v8 ignore next */
}); /* v8 ignore next */ /* v8 ignore next */
if (typeof window !== 'undefined') {
  /* v8 ignore next */ /* v8 ignore next */
  Object.defineProperty(window, 'localStorage', {
    /* v8 ignore next */ /* v8 ignore next */
    value: localStorageMock /* v8 ignore next */ /* v8 ignore next */,
    writable: true /* v8 ignore next */ /* v8 ignore next */,
    configurable: true /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
}
