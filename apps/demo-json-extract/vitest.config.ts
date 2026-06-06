import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "jsdom",
    pool: "forks",
    coverage: {
      provider: "v8",
      include: ["app.ts"],
      reporter: ["text", "json-summary", "json", "html"],
    },
  },
});
