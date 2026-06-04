/* v8 ignore next */ /* v8 ignore next */ import { defineConfig, devices } from '@playwright/test'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export default defineConfig({ /* v8 ignore next */ /* v8 ignore next */
  testDir: './e2e', /* v8 ignore next */ /* v8 ignore next */
  testMatch: 'demo-genai.spec.ts', /* v8 ignore next */ /* v8 ignore next */
  use: { /* v8 ignore next */ /* v8 ignore next */
    baseURL: 'http://localhost:3000', /* v8 ignore next */ /* v8 ignore next */
  }, /* v8 ignore next */ /* v8 ignore next */
  webServer: { /* v8 ignore next */ /* v8 ignore next */
    command: 'cd apps/demo-genai && pnpm preview --port 3000', /* v8 ignore next */ /* v8 ignore next */
    url: 'http://localhost:3000', /* v8 ignore next */ /* v8 ignore next */
  }, /* v8 ignore next */ /* v8 ignore next */
});
