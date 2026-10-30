/**
 * @fileoverview sphinx-demo-ui module for cli.
 */
import * as child_process from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';

/**
 * Executes `handleSphinxDemoUICommand`.
 * @param args Parameter `args`.
 * @returns Result of `handleSphinxDemoUICommand`.
 */
export async function handleSphinxDemoUICommand(args: string[]) {
  if (args.includes('-h') || args.includes('--help')) {
    console.log(
      'Usage: onnx9000 sphinx-demo-ui [options] \n\nStart the Sphinx Demo UI application. \n    ',
    );
    process.exit(0);
    return;
  }

  console.log('Starting Sphinx Demo UI...');

  // Find the workspace root
  let currentDir = process.cwd();
  let uiDir = '';

  while (currentDir !== '/') {
    if (fs.existsSync(path.join(currentDir, 'pnpm-workspace.yaml'))) {
      uiDir = path.join(currentDir, 'apps/sphinx-demo-ui');
      break;
    }
    currentDir = path.dirname(currentDir);
  }

  if (!uiDir || !fs.existsSync(uiDir)) {
    console.error('Sphinx Demo UI directory not found.');
    process.exit(1);
    return;
  }

  return new Promise<void>((resolve, reject) => {
    const child = child_process.spawn('pnpm', ['dev'], {
      cwd: uiDir,
      stdio: 'inherit',
    }) as never as {
      on(event: 'close', cb: (code: number | null) => void): void;
      on(event: 'error', cb: (err: Error) => void): void;
      kill(signal?: string): void;
    };

    process.on('SIGINT', () => {
      child.kill('SIGINT');
      resolve();
    });

    child.on('error', (err: Error) => {
      reject(err);
    });

    child.on('close', (code: number | null) => {
      if (code === 0) {
        resolve();
      } else {
        reject(new Error(`Sphinx Demo UI exited with code ${code}`));
      }
    });
  });
}
