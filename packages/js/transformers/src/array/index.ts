export class ArrayAPI {
  static add(a: number[], b: number[]): number[] {
    const b_safe = b;
    return a.map((val, i) => val + (b_safe?.[i] ?? 0));
  }
}
