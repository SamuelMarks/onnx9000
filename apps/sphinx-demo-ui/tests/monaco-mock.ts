export const editor = {
  create: () => ({
    getValue: () => 'mocked content',
    setValue: () => {},
    layout: () => {},
    dispose: () => {},
    onDidChangeModelContent: () => ({ dispose: () => {} }),
    setModel: () => {}
  }),
  createModel: (content: string) => ({
    getValue: () => content,
    setValue: () => {},
    dispose: () => {}
  }),
  setTheme: () => {}
};
export const Uri = {
  parse: (str: string) => ({ path: str })
};
