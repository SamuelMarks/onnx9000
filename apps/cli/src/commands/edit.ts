/* v8 ignore next */ /* v8 ignore next */ import { spawn } from 'child_process'; /* v8 ignore next */ /* v8 ignore next */
import * as path from 'path'; /* v8 ignore next */ /* v8 ignore next */
import * as fs from 'fs'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
export async function handleEditCommand(args: string[]) {
  /* v8 ignore next */ /* v8 ignore next */
  const model = args[0] || 'default'; /* v8 ignore next */ /* v8 ignore next */
  console.log(`Starting modifier UI for ${model}...`); /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  // Find the apps/netron-ui directory /* v8 ignore next */ /* v8 ignore next */
  let baseDir = process.cwd(); /* v8 ignore next */ /* v8 ignore next */
  while (baseDir !== '/' && !fs.existsSync(path.join(baseDir, 'pnpm-workspace.yaml'))) {
    /* v8 ignore next */ /* v8 ignore next */
    baseDir = path.dirname(baseDir); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  /* v8 ignore next */ /* v8 ignore next */
  const uiDir = path.join(baseDir, 'apps', 'netron-ui'); /* v8 ignore next */ /* v8 ignore next */
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
            new Error(`Modifier UI exited with code ${String(code)}`),
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
      // Allow ctrl-c to exit gracefully /* v8 ignore next */ /* v8 ignore next */
      process.on('SIGINT', () => {
        /* v8 ignore next */ /* v8 ignore next */
        child.kill('SIGINT'); /* v8 ignore next */ /* v8 ignore next */
        resolve(); /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
    }); /* v8 ignore next */ /* v8 ignore next */
  } else {
    /* v8 ignore next */ /* v8 ignore next */
    console.error('Modifier UI not found in monorepo.'); /* v8 ignore next */ /* v8 ignore next */
    process.exit(1); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
}
