/**
 * @fileoverview Convert CLI command handler.
 */
import * as fs from 'node:fs';
import * as path from 'node:path';
import { mmdnn, type SourceFramework, type TargetFramework } from '@onnx9000/converters';

/**
 * Handles the `onnx9000 convert` CLI command.
 * @param args CLI arguments.
 */
export async function handleConvertCommand(args: string[]): Promise<void> {
  let src: SourceFramework | null = null;
  let dst: TargetFramework | null = null;
  const filePaths: string[] = [];

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--src' || arg === '--from') {
      src = (args[++i] ?? null) as SourceFramework | null;
    } else if (arg === '--dst' || arg === '--to') {
      dst = (args[++i] ?? null) as TargetFramework | null;
    } else if (arg && !arg.startsWith('--')) {
      filePaths.push(arg);
    }
  }

  if (!src || !dst) {
    console.error('Usage: onnx9000 convert --src <framework> --dst <framework> <files|directory>');
    process.exit(1);
    return;
  }

  const firstPath = filePaths[0];
  if (!firstPath) {
    console.error('Error: No input files or directory provided.');
    process.exit(1);
    return;
  }

  const stats = fs.statSync(firstPath);
  if (stats.isDirectory()) {
    await handleBatchConversion(firstPath, src, dst);
  } else {
    await processFiles(filePaths, src, dst);
  }
}

/**
 * Executes `handleBatchConversion`.
 * @param dirPath Parameter `dirPath`.
 * @param src Parameter `src`.
 * @param dst Parameter `dst`.
 */
async function handleBatchConversion(
  dirPath: string,
  src: SourceFramework,
  dst: TargetFramework,
): Promise<void> {
  const files = fs.readdirSync(dirPath);
  const fileGroups = new Map<string, string[]>();

  for (const file of files) {
    const fullPath = path.join(dirPath, file);
    if (fs.statSync(fullPath).isFile()) {
      const ext = path.extname(file);
      const base = path.basename(file, ext);
      if (!fileGroups.has(base)) {
        fileGroups.set(base, []);
      }
      fileGroups.get(base)?.push(fullPath);
    }
  }

  console.log(`Starting batch conversion in ${dirPath}`);
  for (const [base, group] of fileGroups) {
    console.log(`Processing group: ${base}`);
    await processFiles(group, src, dst);
  }
}

/**
 * Executes `processFiles`.
 * @param filePaths Parameter `filePaths`.
 * @param src Parameter `src`.
 * @param dst Parameter `dst`.
 */
async function processFiles(
  filePaths: string[],
  src: SourceFramework,
  dst: TargetFramework,
): Promise<void> {
  try {
    console.log(`Converting ${filePaths.join(', ')} from ${src} to ${dst}`);

    const blobs = filePaths.map((p) => {
      const stat = fs.statSync(p);
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
          return {
            size: (end || stat.size) - (start || 0),
          };
        },
      } as never as File;
    });

    const result = await mmdnn.convert(src, dst, blobs, {
      verbose: true,
    });
    console.log(`Conversion completed successfully.`);

    const firstFile = filePaths[0] as string;
    const outName = `${path.basename(firstFile, path.extname(firstFile))}_converted`;
    if (typeof result === 'string') {
      fs.writeFileSync(`${outName}.out`, result);
    } else {
      console.log('Result is an object/graph. Skipping write for now.');
    }
  } catch (e) {
    console.error(`Conversion failed for ${filePaths.join(', ')}:`, e);
  }
}
