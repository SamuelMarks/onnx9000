import tseslint from 'typescript-eslint';
import baseConfig from '../../eslint.config.js';

export default tseslint.config(...baseConfig, {
  rules: {
    '@typescript-eslint/no-explicit-any': 'off',
    '@typescript-eslint/no-unsafe-assignment': 'off',
    '@typescript-eslint/no-unsafe-member-access': 'off',
    '@typescript-eslint/no-unsafe-call': 'off',
    '@typescript-eslint/no-unsafe-return': 'off',
    '@typescript-eslint/no-floating-promises': 'off',
    '@typescript-eslint/restrict-template-expressions': 'off',
    'no-restricted-syntax': 'off',
  },
});
