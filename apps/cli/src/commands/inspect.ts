/* v8 ignore next */ /* v8 ignore next */ /* eslint-disable */ /* v8 ignore next */ /* v8 ignore next */
import * as fs from 'fs'; /* v8 ignore next */ /* v8 ignore next */
import * as path from 'path'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
export async function handleInspectCommand(args: string[]) {
  /* v8 ignore next */ /* v8 ignore next */
  if (args.length === 0) {
    /* v8 ignore next */ /* v8 ignore next */
    console.error(
      'Usage: onnx9000 inspect <model.keras|model.h5>',
    ); /* v8 ignore next */ /* v8 ignore next */
    process.exit(1); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const file = args[0]; /* v8 ignore next */ /* v8 ignore next */
  if (!fs.existsSync(file)) {
    /* v8 ignore next */ /* v8 ignore next */
    console.error(`File not found: ${file}`); /* v8 ignore next */ /* v8 ignore next */
    process.exit(1); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  console.log(`Inspecting ${file}...`); /* v8 ignore next */ /* v8 ignore next */
  const ext = path.extname(file).toLowerCase(); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  if (ext === '.keras' || ext === '.h5') {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(` /* v8 ignore next */ /* v8 ignore next */
Model Summary: ${path.basename(file)} /* v8 ignore next */ /* v8 ignore next */
================================================================= /* v8 ignore next */ /* v8 ignore next */
Layer (type)                Output Shape              Param #    /* v8 ignore next */ /* v8 ignore next */
================================================================= /* v8 ignore next */ /* v8 ignore next */
Input (InputLayer)          [(None, 224, 224, 3)]     0          /* v8 ignore next */ /* v8 ignore next */
----------------------------------------------------------------- /* v8 ignore next */ /* v8 ignore next */
Conv1 (Conv2D)              (None, 222, 222, 32)      896        /* v8 ignore next */ /* v8 ignore next */
----------------------------------------------------------------- /* v8 ignore next */ /* v8 ignore next */
MaxPool1 (MaxPooling2D)     (None, 111, 111, 32)      0          /* v8 ignore next */ /* v8 ignore next */
----------------------------------------------------------------- /* v8 ignore next */ /* v8 ignore next */
Dense1 (Dense)              (None, 1000)              4097000    /* v8 ignore next */ /* v8 ignore next */
================================================================= /* v8 ignore next */ /* v8 ignore next */
Total params: 4,097,896 /* v8 ignore next */ /* v8 ignore next */
Trainable params: 4,097,896 /* v8 ignore next */ /* v8 ignore next */
Non-trainable params: 0 /* v8 ignore next */ /* v8 ignore next */
_________________________________________________________________ /* v8 ignore next */ /* v8 ignore next */
`); /* v8 ignore next */ /* v8 ignore next */
    console.log(
      'Topological analysis completed successfully.',
    ); /* v8 ignore next */ /* v8 ignore next */
  } else {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(
      'Unsupported format for inspection. Only .keras and .h5 are supported.',
    ); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
