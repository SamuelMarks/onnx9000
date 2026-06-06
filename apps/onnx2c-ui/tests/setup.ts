(global as any).Worker = class Worker {
  onmessage: any;
  postMessage(_msg: any) {
    if (this.onmessage) {
      this.onmessage({
        data: {
          header: "mock header",
          source: "mock source",
          summary: "mock summary",
          arenaSize: 500,
        },
      });
    }
  }
};
(global as any).URL.createObjectURL = () => "blob:mock";
(global as any).URL.revokeObjectURL = () => {};
