export function register_op(domain: string, opName: string) {
  return function (target: any) {
    target.domain = domain;
    target.opName = opName;
    return target;
  };
}
