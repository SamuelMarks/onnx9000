import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    rollupOptions: {
      external: ['@onnx9000/tensorrt'],
    },
  },
});
