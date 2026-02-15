import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';
import prettierConfig from 'eslint-config-prettier';

export default tseslint.config(
  // Base ESLint recommended rules
  eslint.configs.recommended,

  // TypeScript ESLint recommended rules
  ...tseslint.configs.recommended,
  ...tseslint.configs.stylistic,

  // Prettier (disables conflicting rules)
  prettierConfig,

  // Custom configuration
  {
    languageOptions: {
      ecmaVersion: 2024,
      sourceType: 'module',
      parserOptions: {
        project: './tsconfig.json',
      },
    },

    rules: {
      // TypeScript specific
      '@typescript-eslint/explicit-function-return-type': 'off',
      '@typescript-eslint/explicit-module-boundary-types': 'off',
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/no-unused-vars': [
        'warn',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
        },
      ],
      '@typescript-eslint/consistent-type-imports': [
        'warn',
        {
          prefer: 'type-imports',
          fixStyle: 'separate-type-imports',
        },
      ],
      '@typescript-eslint/no-import-type-side-effects': 'warn',
      '@typescript-eslint/array-type': 'off', // Allow both T[] and Array<T>
      '@typescript-eslint/no-inferrable-types': 'off', // Allow explicit types
      '@typescript-eslint/no-empty-function': 'off', // Allow empty functions
      '@typescript-eslint/consistent-generic-constructors': 'off',

      // General
      'no-console': 'off', // Allowed for CLI app
      'prefer-const': 'error',
      'no-var': 'error',
      'eqeqeq': ['error', 'always'],
      'curly': 'off', // Allow single-line if without braces
      'no-throw-literal': 'error',

      // Import ordering - disabled for now (too many changes)
      'sort-imports': 'off',
    },
  },

  // Ignore patterns
  {
    ignores: [
      'dist/**',
      'node_modules/**',
      'web/**',
      'data/**',
      '*.js',
      '!eslint.config.js',
      'src/skills/executor/skill-worker.js',
      'crew/python/**',
    ],
  }
);
