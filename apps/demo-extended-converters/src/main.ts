/**
 * @fileoverview Extended Converters Demo UI.
 */
import { mmdnn, type SourceFramework, type TargetFramework } from '@onnx9000/converters';

/**
 * Initializes the extended converters demo.
 */
export function initExtendedConvertersDemo(): void {
  const btnConvert = document.getElementById('btnConvert') as HTMLButtonElement | null;
  const output = document.getElementById('output') as HTMLDivElement | null;
  const fileInput = document.getElementById('fileInput') as HTMLInputElement | null;
  const srcSelect = document.getElementById('srcFramework') as HTMLSelectElement | null;
  const dstSelect = document.getElementById('dstFramework') as HTMLSelectElement | null;

  if (!btnConvert || !output || !fileInput || !srcSelect || !dstSelect) return;

  btnConvert.addEventListener('click', async () => {
    if (!fileInput.files || fileInput.files.length === 0) {
      output.textContent = 'Please select one or more files to convert.';
      return;
    }

    const files = Array.from(fileInput.files);
    const src = srcSelect.value as SourceFramework;
    const dst = dstSelect.value as TargetFramework;

    output.textContent = `Converting ${files.length} file(s) from ${src} to ${dst}...

`;

    try {
      const result = await mmdnn.convert(src, dst, files, {
        verbose: true,
      });

      output.textContent += `Conversion Successful!

`;

      if (typeof result === 'string') {
        output.textContent += `Result Type: String Payload
Length: ${result.length}`;
      } else {
        output.textContent += `Result Type: Graph Object
${JSON.stringify(result, null, 2).slice(0, 500)}...`;
      }
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      output.textContent += `Error during conversion: ${msg}`;
    }
  });
}

initExtendedConvertersDemo();
