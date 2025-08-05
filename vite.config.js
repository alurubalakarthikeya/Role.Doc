import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: '/RoleDoc/',
  build: {
    outDir: 'dist',
  },
  server: {
    proxy: {
      '/query': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
    },
  }
});
