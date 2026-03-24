/**
 * A simple utility to debounce function calls.
 */
export class Debouncer {
  private timeoutId: ReturnType<typeof setTimeout> | null = null;

  /**
   * Debounce a function.
   *
   * @param func - The function to debounce.
   * @param delay - The delay in milliseconds.
   * @returns A wrapped, debounced function.
   */
  public debounce<T extends (...args: any[]) => void>(
    func: T,
    delay: number
  ): (...args: Parameters<T>) => void {
    return (...args: Parameters<T>) => {
      if (this.timeoutId) {
        clearTimeout(this.timeoutId);
      }
      this.timeoutId = setTimeout(() => {
        func(...args);
      }, delay);
    };
  }

  /**
   * Immediately clears any pending debounced calls.
   */
  public clear(): void {
    if (this.timeoutId) {
      clearTimeout(this.timeoutId);
      this.timeoutId = null;
    }
  }
}
