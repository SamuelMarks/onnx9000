(global as typeof globalThis & { Worker: unknown }).Worker = class Worker {
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  postMessage(_msg: unknown) {
    if (this.onmessage) {
      this.onmessage({
        data: {
          header: 'mock header',
          source: 'mock source',
          summary: 'mock summary',
          arenaSize: 500,
        },
      });
    }
  }
};
global.URL.createObjectURL = () => 'blob:mock';
global.URL.revokeObjectURL = () => {};
