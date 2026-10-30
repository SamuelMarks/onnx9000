export default [
  {
    extends: './vitest.config.ts',
    test: {
      name: 'onnx9000',
      include: ['packages/js/*/tests/**/*.test.ts', 'apps/*/tests/**/*.test.ts'],
    },
  },
];
