import { spawn } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

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
    const child = spawn('pnpm', ['dev'], { cwd: uiDir, stdio: 'inherit' });

    return new Promise<void>((resolve, reject) => {
      child.on('close', (code) => {
        if (code !== 0 && code !== null) {
          reject(new Error(`Modifier UI exited with code ${String(code)}`));
        } else {
          resolve();
        }
      });
      child.on('error', (err) => {
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
