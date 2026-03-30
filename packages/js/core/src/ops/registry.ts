export function register_op(domain: string, opName: string) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return function (target: any) {
    target.domain = domain;
    target.opName = opName;
    return target;
  };
}
