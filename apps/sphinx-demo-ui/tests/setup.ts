export {};

const store: Record<string, string> = {};
const mockStorage = {
  getItem: (key: string) => store[key] ?? null,
  setItem: (key: string, val: string) => {
    store[key] = String(val);
  },
  removeItem: (key: string) => {
    delete store[key];
  },
  clear: () => {
    for (const k of Object.keys(store)) delete store[k];
  },
};

Object.defineProperty(globalThis, 'localStorage', {
  value: mockStorage,
  configurable: true,
  writable: true,
});

if (typeof window !== 'undefined') {
  Object.defineProperty(window, 'localStorage', {
    value: mockStorage,
    configurable: true,
    writable: true,
  });
}

(globalThis as Record<string, unknown>).Worker = class Worker {
  postMessage() {}
  onmessage() {}
};
(globalThis as Record<string, unknown>).ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};
