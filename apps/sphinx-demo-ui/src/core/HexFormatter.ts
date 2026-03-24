/**
 * Utility to format binary ArrayBuffers into readable Hex strings
 * for the Monaco editor when a target is a pure binary file (like .onnx or .pb).
 */
export class HexFormatter {
  /**
   * Formats a Uint8Array into a hex dump format.
   *
   * @param data The binary data to format.
   * @param maxLength Maximum number of bytes to format to avoid freezing the editor.
   * @returns The formatted hex string.
   */
  public static format(data: Uint8Array, maxLength: number = 4096): string {
    if (data.length === 0) return '';

    let result = '';
    const length = Math.min(data.length, maxLength);

    for (let i = 0; i < length; i += 16) {
      const chunk = data.subarray(i, Math.min(i + 16, length));

      // Offset address
      result += i.toString(16).padStart(8, '0') + '  ';

      // Hex representation
      let hexPart1 = '';
      let hexPart2 = '';
      let asciiPart = '';

      for (let j = 0; j < 16; j++) {
        if (j < chunk.length) {
          const byte = chunk[j];
          const hex = byte.toString(16).padStart(2, '0');
          if (j < 8) {
            hexPart1 += hex + ' ';
          } else {
            hexPart2 += hex + ' ';
          }

          // Printable ASCII or dot
          if (byte >= 32 && byte <= 126) {
            asciiPart += String.fromCharCode(byte);
          } else {
            asciiPart += '.';
          }
        } else {
          // Padding
          if (j < 8) {
            hexPart1 += '   ';
          } else {
            hexPart2 += '   ';
          }
          asciiPart += ' ';
        }
      }

      result += hexPart1 + ' ' + hexPart2 + ' |' + asciiPart.replace(/\s+$/, '') + '|\n';
    }

    if (data.length > maxLength) {
      result += `\n... (truncated, displaying first ${maxLength} bytes of ${data.length} total) ...\n`;
    }

    return result.trimEnd();
  }
}
