import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  build: {
    lib: {
      entry: 'src/main.ts',
      name: 'SphinxDemoUI',
      fileName: 'sphinx-demo-ui',
      formats: ['es', 'umd']
    },
    outDir: 'dist',
    rollupOptions: { external: ['@onnx9000/webnn-polyfill'] }
  },
  optimizeDeps: {
    exclude: ['pyodide']
  },
  test: {
    environment: 'jsdom',
    alias: [{ find: /^monaco-editor.*/, replacement: resolve(__dirname, 'tests/monaco-mock.ts') }],
    globals: true,
    setupFiles: ['./tests/setup.ts'],
    include: ['tests/**/*.test.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      lines: 100,
      functions: 100,
      branches: 100,
      statements: 100,
      exclude: ['e2e/**/*', 'src/main.ts', 'src/types/**/*', 'docs/**/*']
    }
  }
});
