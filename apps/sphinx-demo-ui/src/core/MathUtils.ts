export class MathUtils {
  /**
   * Calculates Mean.
   */
  public static mean(data: number[]): number {
    if (data.length === 0) return 0;
    const sum = data.reduce((acc, val) => acc + val, 0);
    return sum / data.length;
  }

  /**
   * Calculates Variance.
   */
  public static variance(data: number[]): number {
    if (data.length === 0) return 0;
    const m = this.mean(data);
    const sumSquareDiff = data.reduce((acc, val) => acc + Math.pow(val - m, 2), 0);
    return sumSquareDiff / data.length;
  }

  /**
   * Normalizes data for charting between 0 and 1.
   */
  public static normalize(data: number[]): number[] {
    if (data.length === 0) return [];
    const min = Math.min(...data);
    const max = Math.max(...data);
    if (max - min === 0) return data.map(() => 0.5); // Flat line
    return data.map((v) => (v - min) / (max - min));
  }
}
