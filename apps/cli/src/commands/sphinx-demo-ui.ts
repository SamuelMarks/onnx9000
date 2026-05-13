export async function handleSphinxDemoUICommand(args: string[]) {
  if (args.includes('-h') || args.includes('--help')) {
    console.log(`Usage: onnx9000 sphinx-demo-ui [options]

Start the Sphinx Demo UI application.
    `);
    process.exit(0);
    return;
  }

  console.log('Starting Sphinx Demo UI...');

  // Find the apps/sphinx-demo-ui directory
  const path = await import('path');
  const fs = await import('fs');
  const { spawn } = await import('child_process');

  let baseDir = process.cwd();
  while (baseDir !== '/' && !fs.existsSync(path.join(baseDir, 'pnpm-workspace.yaml'))) {
    baseDir = path.dirname(baseDir);
  }

  const uiDir = path.join(baseDir, 'apps', 'sphinx-demo-ui');
  if (fs.existsSync(uiDir)) {
    console.log(`Opening ${uiDir}...`);
    const child = spawn('pnpm', ['dev'], { cwd: uiDir, stdio: 'inherit' });

    return new Promise<void>((resolve, reject) => {
      child.on('close', (code) => {
        if (code !== 0 && code !== null) {
          reject(new Error(`Sphinx Demo UI exited with code ${String(code)}`));
        } else {
          resolve();
        }
      });
      child.on('error', (err) => {
        reject(err);
      });
      process.on('SIGINT', () => {
        child.kill('SIGINT');
        resolve();
      });
    });
  } else {
    console.error('Sphinx Demo UI not found in monorepo.');
    process.exit(1);
  }
}
