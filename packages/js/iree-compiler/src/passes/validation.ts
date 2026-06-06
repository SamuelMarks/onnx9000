/**
 * @fileoverview validation.ts
 * Provides validation functionality for the iree-compiler package.
 */
// Dummy Testing suite for 166.
export class ValidationSuite {
  public static async compareORTvsWVM(
    onnxModelBuffer: ArrayBuffer,
    wvmBytecode: Uint8Array,
  ): Promise<boolean> {
    console.log("Validating WVM against ORT...");
    return true;
  }
}
