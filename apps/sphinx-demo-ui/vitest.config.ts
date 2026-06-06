import { resolve } from 'node:path';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./tests/setup.ts'],
    exclude: ['e2e/**', 'node_modules/**'],
    alias: [
      {
        find: /^monaco-editor.*/,
        replacement: resolve(__dirname, './tests/__mocks__/monaco-editor.ts'),
      },
    ],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      include: ['src/**/*.ts'],
    },
  },
});
