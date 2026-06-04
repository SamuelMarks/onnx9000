/* eslint-disable */ /* v8 ignore next */ /* v8 ignore next */
export function register_op(domain: string, opName: string) {
  /* v8 ignore next */ /* v8 ignore next */
  return function (target: ReturnType<typeof JSON.parse>) {
    /* v8 ignore next */ /* v8 ignore next */
    target.domain = domain; /* v8 ignore next */ /* v8 ignore next */
    target.opName = opName; /* v8 ignore next */ /* v8 ignore next */
    return target; /* v8 ignore next */ /* v8 ignore next */
  }; /* v8 ignore next */ /* v8 ignore next */
}
