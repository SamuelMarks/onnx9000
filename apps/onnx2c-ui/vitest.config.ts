import { resolve } from "node:path";
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "jsdom",
    pool: "forks",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
    alias: {
      "monaco-editor": resolve(__dirname, "./tests/__mocks__/monaco-editor.ts"),
    },
    coverage: {
      provider: "v8",
      reporter: ["text", "json", "html"],
      include: ["src/**/*.ts"],
    },
  },
});
