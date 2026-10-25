import { svelte } from '@sveltejs/vite-plugin-svelte'
import { defineConfig } from 'vite'

const host = process.env.TAURI_DEV_HOST;

// https://v2.tauri.app/start/frontend/vite/
export default defineConfig({
  plugins: [svelte()],
  clearScreen: false,
  server: {
    port: 1440,
    strictPort: true,
    host: host || false,
    hmr: host ? { protocol: 'ws', host, port: 1441 } : undefined,
    watch: { ignored: ['**/core/**'] },
  },
  // Modern WebViews only: no syntax down-levelling, less code to parse at start.
  build: {
    target: ['es2022', 'chrome105'],
  },
})
