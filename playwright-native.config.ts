import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './e2e',
  testMatch: 'demo-native.spec.ts',
  use: {
    baseURL: 'http://localhost:3000',
  },
  webServer: {
    command: 'cd apps/demo-native && pnpm preview --port 3000',
    url: 'http://localhost:3000',
  },
});
