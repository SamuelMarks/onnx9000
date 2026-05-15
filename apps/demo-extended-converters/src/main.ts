import { mmdnn } from '@onnx9000/converters';

const btnConvert = document.getElementById('btnConvert') as HTMLButtonElement;
const output = document.getElementById('output') as HTMLDivElement;
const fileInput = document.getElementById('fileInput') as HTMLInputElement;
const srcSelect = document.getElementById('srcFramework') as HTMLSelectElement;
const dstSelect = document.getElementById('dstFramework') as HTMLSelectElement;

btnConvert.addEventListener('click', async () => {
  if (!fileInput.files || fileInput.files.length === 0) {
    output.textContent = 'Please select one or more files to convert.';
    return;
  }

  const files = Array.from(fileInput.files);
  const src = srcSelect.value;
  const dst = dstSelect.value;

  output.textContent = `Converting ${files.length} file(s) from ${src} to ${dst}...\n\n`;

  try {
    // Attempt conversion using the Web/Browser API
    const result = await mmdnn.convert(src as any, dst as any, files as any, { verbose: true });

    output.textContent += `Conversion Successful!\n\n`;

    // Minimal display of returned output
    if (typeof result === 'string') {
      output.textContent += `Result Type: String Payload\nLength: ${result.length}`;
    } else {
      output.textContent += `Result Type: Graph Object\n${JSON.stringify(result, null, 2).slice(0, 500)}...`;
    }
  } catch (error: any) {
    output.textContent += `Error during conversion: ${error.message}`;
  }
});
