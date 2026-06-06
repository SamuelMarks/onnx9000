(global as any).localStorage = {
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {},
  clear: () => {},
};
(global as any).Worker = class Worker {
  postMessage() {}
  onmessage() {}
};
(global as any).ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
};
