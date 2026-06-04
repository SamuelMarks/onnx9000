import { defineConfig } from 'vitest/config';
export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    exclude: ['**/node_modules/**', '**/dist/**', '**/e2e/**', '**/test_out/**', '**/.git/**', 'apps/**', 'packages/python/**', 'tests/**'],
    coverage: {
      provider: 'v8',
      include: ['packages/js/core/src/ir/node.ts'],
      thresholds: { lines: 100, functions: 100, branches: 100, statements: 100 }
    }
  }
});
