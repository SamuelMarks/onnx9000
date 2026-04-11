/* eslint-disable */
export function register_op(domain: string, opName: string) {
  return function (target: ReturnType<typeof JSON.parse>) {
    target.domain = domain;
    target.opName = opName;
    return target;
  };
}
