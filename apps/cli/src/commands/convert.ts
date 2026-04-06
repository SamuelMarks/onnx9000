import * as fs from 'fs';
import * as path from 'path';
import { mmdnn } from '@onnx9000/converters';
const { convert } = mmdnn;
type SourceFramework = Object;
type TargetFramework = Object;

export async function handleConvertCommand(args: string[]) {
  let src: SourceFramework | null = null;
  let dst: TargetFramework | null = null;
  const filePaths: string[] = [];

  // Parse arguments
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--src' || arg === '--from') {
      src = args[++i] as SourceFramework;
    } else if (arg === '--dst' || arg === '--to') {
      dst = args[++i] as TargetFramework;
    } else if (!arg.startsWith('--')) {
      filePaths.push(arg);
    }
  }

  if (!src || !dst) {
    console.error('Usage: onnx9000 convert --src <framework> --dst <framework> <files|directory>');
    process.exit(1);
  }

  if (filePaths.length === 0) {
    console.error('Error: No input files or directory provided.');
    process.exit(1);
  }

  const stats = fs.statSync(filePaths[0]);
  if (stats.isDirectory()) {
    await handleBatchConversion(filePaths[0], src, dst);
  } else {
    await processFiles(filePaths, src, dst);
  }
}

async function handleBatchConversion(dirPath: string, src: SourceFramework, dst: TargetFramework) {
  const files = fs.readdirSync(dirPath);
  // Group files by base name for frameworks that need multiple files (like caffe with prototxt/caffemodel)
  const fileGroups = new Map<string, string[]>();

  for (const file of files) {
    const fullPath = path.join(dirPath, file);
    if (fs.statSync(fullPath).isFile()) {
      const ext = path.extname(file);
      const base = path.basename(file, ext);
      if (!fileGroups.has(base)) {
        fileGroups.set(base, []);
      }
      fileGroups.get(base)!.push(fullPath);
    }
  }

  console.log(`Starting batch conversion in ${dirPath}`);
  for (const [base, group] of fileGroups) {
    console.log(`Processing group: ${base}`);
    await processFiles(group, src, dst);
  }
}

async function processFiles(filePaths: string[], src: SourceFramework, dst: TargetFramework) {
  try {
    console.log(`Converting ${filePaths.join(', ')} from ${src} to ${dst}`);

    // Handle massive file conversions via streaming buffers in Node.js to avoid Heap exhaustion.
    // Instead of reading the whole file into a Buffer, we pass node streams wrapped as web Blobs/Files
    // (Assuming the underlying API supports this or we simulate it here)
    const blobs = filePaths.map((p) => {
      const stat = fs.statSync(p);
      // We use a Blob-like object that streams from the file
      return {
        size: stat.size,
        type: 'application/octet-stream',
        name: path.basename(p),
        stream: () => fs.createReadStream(p),
        arrayBuffer: async () => {
          const buf = await fs.promises.readFile(p);
          return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
        },
        slice: (start?: number, end?: number) => {
          // simulate slice for memory-mapped chunking
          return { size: (end || stat.size) - (start || 0) };
        },
      };
    });

    // @ts-ignore
    const result = await convert(src, dst, blobs, { verbose: true });
    console.log(`Conversion completed successfully.`);

    // Simulating write out based on framework
    const outName = `${path.basename(filePaths[0], path.extname(filePaths[0]))}_converted`;
    if (typeof result === 'string') {
      fs.writeFileSync(`${outName}.out`, result);
    } else {
      console.log('Result is an object/graph. Skipping write for now.');
    }
  } catch (e) {
    console.error(`Conversion failed for ${filePaths.join(', ')}:`, e);
  }
}
