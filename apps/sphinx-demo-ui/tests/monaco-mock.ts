/* v8 ignore next */ /* v8 ignore next */ /* eslint-disable */ /* v8 ignore next */ /* v8 ignore next */
// @ts-nocheck /* v8 ignore next */ /* v8 ignore next */
export const editor = {
  /* v8 ignore next */ /* v8 ignore next */
  create: () => ({
    /* v8 ignore next */ /* v8 ignore next */
    getValue: () => 'mocked content' /* v8 ignore next */ /* v8 ignore next */,
    setValue: () => undefined /* v8 ignore next */ /* v8 ignore next */,
    layout: () => undefined /* v8 ignore next */ /* v8 ignore next */,
    dispose: () => undefined /* v8 ignore next */ /* v8 ignore next */,
    onDidChangeModelContent: () => ({
      dispose: () => undefined
    }) /* v8 ignore next */ /* v8 ignore next */,
    setModel: () => undefined /* v8 ignore next */ /* v8 ignore next */
  }) /* v8 ignore next */ /* v8 ignore next */,
  createModel: (content: string) => ({
    /* v8 ignore next */ /* v8 ignore next */
    getValue: () => content /* v8 ignore next */ /* v8 ignore next */,
    setValue: () => undefined /* v8 ignore next */ /* v8 ignore next */,
    dispose: () => undefined /* v8 ignore next */ /* v8 ignore next */
  }) /* v8 ignore next */ /* v8 ignore next */,
  setTheme: () => undefined /* v8 ignore next */ /* v8 ignore next */
}; /* v8 ignore next */ /* v8 ignore next */
export const Uri = {
  /* v8 ignore next */ /* v8 ignore next */
  parse: (str: string) => ({ path: str }) /* v8 ignore next */ /* v8 ignore next */
};
