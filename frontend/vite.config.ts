import { fileURLToPath, URL } from 'node:url'

import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'

// 开发期把后端相关路径全部代理到 FastAPI，避免前端出现硬编码域名与 CORS 问题。
// 生产构建时前端由 FastAPI 静态托管，同源访问，无需代理。
const BACKEND = 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api': { target: BACKEND, changeOrigin: true },
      '/static': { target: BACKEND, changeOrigin: true },
      '/uploads': { target: BACKEND, changeOrigin: true },
      '/figures': { target: BACKEND, changeOrigin: true },
      '/healthz': { target: BACKEND, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    chunkSizeWarningLimit: 1500,
  },
})
