/**
 * @fileoverview edit module for cli.
 */
import * as child_process from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';

/**
 * Executes `handleEditCommand`.
 * @param args Parameter `args`.
 * @returns Result of `handleEditCommand`.
 */
export async function handleEditCommand(args: string[]) {
  const model = args[0] || 'default';
  console.log(`Starting modifier UI for ${model}...`);

  // Find the apps/netron-ui directory
  let baseDir = process.cwd();
  while (baseDir !== '/' && !fs.existsSync(path.join(baseDir, 'pnpm-workspace.yaml'))) {
    baseDir = path.dirname(baseDir);
  }

  const uiDir = path.join(baseDir, 'apps', 'netron-ui');
  if (fs.existsSync(uiDir)) {
    console.log(`Opening ${uiDir}...`);
    const child = child_process.spawn('pnpm', ['dev'], {
      cwd: uiDir,
      stdio: 'inherit',
    }) as never as {
      on(event: 'close', cb: (code: number | null) => void): void;
      on(event: 'error', cb: (err: Error) => void): void;
      kill(signal?: string): void;
    };

    return new Promise<void>((resolve, reject) => {
      child.on('close', (code: number | null) => {
        if (code !== 0 && code !== null) {
          reject(new Error(`Modifier UI exited with code ${String(code)}`));
        } else {
          resolve();
        }
      });
      child.on('error', (err: Error) => {
        reject(err);
      });
      // Allow ctrl-c to exit gracefully
      process.on('SIGINT', () => {
        child.kill('SIGINT');
        resolve();
      });
    });
  } else {
    console.error('Modifier UI not found in monorepo.');
    process.exit(1);
  }
}
