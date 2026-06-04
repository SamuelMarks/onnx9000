/* v8 ignore next */ /* v8 ignore next */ import {
  defineConfig,
  devices
} from '@playwright/test'; /* v8 ignore next */ /* v8 ignore next */
/* v8 ignore next */ /* v8 ignore next */
export default defineConfig({
  /* v8 ignore next */ /* v8 ignore next */
  testDir: './e2e' /* v8 ignore next */ /* v8 ignore next */,
  fullyParallel: true /* v8 ignore next */ /* v8 ignore next */,
  forbidOnly: !!process.env.CI /* v8 ignore next */ /* v8 ignore next */,
  retries: process.env.CI ? 2 : 0 /* v8 ignore next */ /* v8 ignore next */,
  workers: process.env.CI ? 1 : undefined /* v8 ignore next */ /* v8 ignore next */,
  reporter: [['html', { open: 'never' }]] /* v8 ignore next */ /* v8 ignore next */,
  use: {
    /* v8 ignore next */ /* v8 ignore next */
    trace: 'on-first-retry' /* v8 ignore next */ /* v8 ignore next */,
    baseURL: 'http://localhost:5173' /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */,
  projects: [
    /* v8 ignore next */ /* v8 ignore next */
    {
      /* v8 ignore next */ /* v8 ignore next */
      name: 'chromium' /* v8 ignore next */ /* v8 ignore next */,
      use: { ...devices['Desktop Chrome'] } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  ] /* v8 ignore next */ /* v8 ignore next */,
  webServer: {
    /* v8 ignore next */ /* v8 ignore next */
    command: 'npx vite --host localhost --port 5173' /* v8 ignore next */ /* v8 ignore next */,
    url: 'http://localhost:5173' /* v8 ignore next */ /* v8 ignore next */,
    reuseExistingServer: !process.env.CI /* v8 ignore next */ /* v8 ignore next */,
    timeout: 120 * 1000 /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
});
