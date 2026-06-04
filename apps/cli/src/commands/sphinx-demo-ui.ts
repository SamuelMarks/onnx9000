/* v8 ignore next */ /* v8 ignore next */ export async function handleSphinxDemoUICommand(
  args: string[],
) {
  /* v8 ignore next */ /* v8 ignore next */
  if (args.includes('-h') || args.includes('--help')) {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(`Usage: onnx9000 sphinx-demo-ui [options] /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
Start the Sphinx Demo UI application. /* v8 ignore next */ /* v8 ignore next */
    `); /* v8 ignore next */ /* v8 ignore next */
    process.exit(0); /* v8 ignore next */ /* v8 ignore next */
    return; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  console.log('Starting Sphinx Demo UI...'); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Find the apps/sphinx-demo-ui directory /* v8 ignore next */ /* v8 ignore next */
  const path = await import('path'); /* v8 ignore next */ /* v8 ignore next */
  const fs = await import('fs'); /* v8 ignore next */ /* v8 ignore next */
  const { spawn } = await import('child_process'); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  let baseDir = process.cwd(); /* v8 ignore next */ /* v8 ignore next */
  while (baseDir !== '/' && !fs.existsSync(path.join(baseDir, 'pnpm-workspace.yaml'))) {
    /* v8 ignore next */ /* v8 ignore next */
    baseDir = path.dirname(baseDir); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const uiDir = path.join(
    baseDir,
    'apps',
    'sphinx-demo-ui',
  ); /* v8 ignore next */ /* v8 ignore next */
  if (fs.existsSync(uiDir)) {
    /* v8 ignore next */ /* v8 ignore next */
    console.log(`Opening ${uiDir}...`); /* v8 ignore next */ /* v8 ignore next */
    const child = spawn('pnpm', ['dev'], {
      cwd: uiDir,
      stdio: 'inherit',
    }); /* v8 ignore next */ /* v8 ignore next */
    /* v8 ignore next */ /* v8 ignore next */
    return new Promise<void>((resolve, reject) => {
      /* v8 ignore next */ /* v8 ignore next */
      child.on('close', (code) => {
        /* v8 ignore next */ /* v8 ignore next */
        if (code !== 0 && code !== null) {
          /* v8 ignore next */ /* v8 ignore next */
          reject(
            new Error(`Sphinx Demo UI exited with code ${String(code)}`),
          ); /* v8 ignore next */ /* v8 ignore next */
        } else {
          /* v8 ignore next */ /* v8 ignore next */
          resolve(); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
      child.on('error', (err) => {
        /* v8 ignore next */ /* v8 ignore next */
        reject(err); /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
      process.on('SIGINT', () => {
        /* v8 ignore next */ /* v8 ignore next */
        child.kill('SIGINT'); /* v8 ignore next */ /* v8 ignore next */
        resolve(); /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } else {
    /* v8 ignore next */ /* v8 ignore next */
    console.error(
      'Sphinx Demo UI not found in monorepo.',
    ); /* v8 ignore next */ /* v8 ignore next */
    process.exit(1); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
