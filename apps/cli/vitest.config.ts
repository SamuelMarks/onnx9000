import { defineConfig } from 'vitest/config';
export default defineConfig({
  test: { globals: true, coverage: { include: ['apps/cli/src/commands/*.ts'] } },
});
