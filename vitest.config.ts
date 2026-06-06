import { defineConfig } from 'vitest/config';
export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    exclude: [
      '**/node_modules/**',
      '**/dist/**',
      '**/e2e/**',
      '**/test_out/**',
      '**/.git/**',
      'apps/**',
      'packages/python/**',
    ],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json-summary', 'json', 'html'],
    },
  },
});
