import { defineConfig } from 'vite';
export default defineConfig({
  server: { port: 3000 },
  optimizeDeps: { exclude: ['pyodide', '@onnx9000/c-compiler'] },
  worker: {
    format: 'es',
  },
});
