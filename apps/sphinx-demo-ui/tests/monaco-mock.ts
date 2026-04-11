/* eslint-disable */
// @ts-nocheck
export const editor = {
  create: () => ({
    getValue: () => 'mocked content',
    setValue: () => undefined,
    layout: () => undefined,
    dispose: () => undefined,
    onDidChangeModelContent: () => ({ dispose: () => undefined }),
    setModel: () => undefined
  }),
  createModel: (content: string) => ({
    getValue: () => content,
    setValue: () => undefined,
    dispose: () => undefined
  }),
  setTheme: () => undefined
};
export const Uri = {
  parse: (str: string) => ({ path: str })
};
