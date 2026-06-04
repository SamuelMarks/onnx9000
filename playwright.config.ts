/* v8 ignore next */ /* v8 ignore next */ import { defineConfig, devices } from '@playwright/test'; /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
export default defineConfig({ /* v8 ignore next */ /* v8 ignore next */
  testDir: './e2e', /* v8 ignore next */ /* v8 ignore next */
  fullyParallel: true, /* v8 ignore next */ /* v8 ignore next */
  use: { /* v8 ignore next */ /* v8 ignore next */
    baseURL: 'http://localhost:8000', /* v8 ignore next */ /* v8 ignore next */
  }, /* v8 ignore next */ /* v8 ignore next */
  projects: [ /* v8 ignore next */ /* v8 ignore next */
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } }, /* v8 ignore next */ /* v8 ignore next */
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } }, /* v8 ignore next */ /* v8 ignore next */
    { name: 'webkit', use: { ...devices['Desktop Safari'] } }, /* v8 ignore next */ /* v8 ignore next */
    { name: 'Mobile Safari', use: { ...devices['iPhone 12'] } }, /* v8 ignore next */ /* v8 ignore next */
  ], /* v8 ignore next */ /* v8 ignore next */
  webServer: { /* v8 ignore next */ /* v8 ignore next */
    command: 'cd docs && make html && python3 -m http.server 8000 -d _build/html', /* v8 ignore next */ /* v8 ignore next */
    url: 'http://localhost:8000', /* v8 ignore next */ /* v8 ignore next */
    reuseExistingServer: !process.env.CI, /* v8 ignore next */ /* v8 ignore next */
    timeout: 120 * 1000, /* v8 ignore next */ /* v8 ignore next */
  }, /* v8 ignore next */ /* v8 ignore next */
});
