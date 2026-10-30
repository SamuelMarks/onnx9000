import path from 'node:path';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  resolve: {
    alias: [
      {
        find: /^vitest$/,
        replacement: path.resolve(__dirname, 'node_modules/vitest/dist/index.js'),
      },
      {
        find: /^monaco-editor.*/,
        replacement: path.resolve(
          __dirname,
          'apps/sphinx-demo-ui/tests/__mocks__/monaco-editor.ts',
        ),
      },
      {
        find: /^@onnx9000\/openvino-exporter$/,
        replacement: path.resolve(__dirname, 'packages/js/openvino/src/index.ts'),
      },
      {
        find: /^@onnx9000\/([^/]+)$/,
        replacement: path.resolve(__dirname, 'packages/js/$1/src/index.ts'),
      },
      {
        find: /^@onnx9000\/([^/]+)\/dist\/(.*)$/,
        replacement: path.resolve(__dirname, 'packages/js/$1/src/$2'),
      },
      {
        find: /^@onnx9000\/([^/]+)\/src\/(.*)$/,
        replacement: path.resolve(__dirname, 'packages/js/$1/src/$2'),
      },
      {
        find: /^@onnx9000\/([^/]+)\/(.*)$/,
        replacement: path.resolve(__dirname, 'packages/js/$1/src/$2'),
      },
    ],
    dedupe: ['vitest'],
  },
  test: {
    globals: true,
    passWithNoTests: true,
    environment: 'jsdom',
    setupFiles: ['./apps/sphinx-demo-ui/tests/setup.ts'],
    exclude: [
      '**/node_modules/**',
      '**/dist/**',
      '**/e2e/**',
      '**/test_out/**',
      '**/.git/**',
      'packages/python/**',
    ],
    coverage: {
      provider: 'v8',
      all: true,
      include: ['packages/js/*/src/**/*.ts', 'apps/*/src/**/*.ts'],
      exclude: ['**/*.d.ts', '**/dist/**', '**/node_modules/**'],
      reporter: ['text', 'json-summary', 'json', 'html'],
      thresholds: {
        lines: 100,
        functions: 100,
        branches: 100,
        statements: 100,
      },
    },
  },
});
