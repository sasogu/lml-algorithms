import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    setupFiles: ['./tests/setup/vitest.setup.js'],
    clearMocks: true,
    restoreMocks: true,
    include: ['tests/**/*.test.js'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      lines: 70,
      branches: 65,
      functions: 70,
      statements: 70
    }
  }
});
