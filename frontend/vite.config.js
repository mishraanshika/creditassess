import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3030,
    proxy: {
      // Everything the API owns is proxied so the app runs same-origin in dev.
      '/api': { target: 'http://127.0.0.1:8050', changeOrigin: true, rewrite: p => p.replace(/^\/api/, '') }
    }
  }
})
