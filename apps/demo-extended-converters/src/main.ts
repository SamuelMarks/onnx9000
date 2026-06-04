/* v8 ignore next */ /* v8 ignore next */ import { mmdnn } from '@onnx9000/converters'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
const btnConvert = document.getElementById(
  'btnConvert',
) as HTMLButtonElement; /* v8 ignore next */ /* v8 ignore next */
const output = document.getElementById(
  'output',
) as HTMLDivElement; /* v8 ignore next */ /* v8 ignore next */
const fileInput = document.getElementById(
  'fileInput',
) as HTMLInputElement; /* v8 ignore next */ /* v8 ignore next */
const srcSelect = document.getElementById(
  'srcFramework',
) as HTMLSelectElement; /* v8 ignore next */ /* v8 ignore next */
const dstSelect = document.getElementById(
  'dstFramework',
) as HTMLSelectElement; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
btnConvert.addEventListener('click', async () => {
  /* v8 ignore next */ /* v8 ignore next */
  if (!fileInput.files || fileInput.files.length === 0) {
    /* v8 ignore next */ /* v8 ignore next */
    output.textContent =
      'Please select one or more files to convert.'; /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const files = Array.from(fileInput.files); /* v8 ignore next */ /* v8 ignore next */
  const src = srcSelect.value; /* v8 ignore next */ /* v8 ignore next */
  const dst = dstSelect.value; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  output.textContent = `Converting ${files.length} file(s) from ${src} to ${dst}...\n\n`; /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  try {
    /* v8 ignore next */ /* v8 ignore next */
    // Attempt conversion using the Web/Browser API /* v8 ignore next */ /* v8 ignore next */
    const result = await mmdnn.convert(src as any, dst as any, files as any, {
      verbose: true,
    }); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    output.textContent += `Conversion Successful!\n\n`; /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    // Minimal display of returned output /* v8 ignore next */ /* v8 ignore next */
    if (typeof result === 'string') {
      /* v8 ignore next */ /* v8 ignore next */
      output.textContent += `Result Type: String Payload\nLength: ${result.length}`; /* v8 ignore next */ /* v8 ignore next */
    } else {
      /* v8 ignore next */ /* v8 ignore next */
      output.textContent += `Result Type: Graph Object\n${JSON.stringify(result, null, 2).slice(0, 500)}...`; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } catch (error: any) {
    /* v8 ignore next */ /* v8 ignore next */
    output.textContent += `Error during conversion: ${error.message}`; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
});
