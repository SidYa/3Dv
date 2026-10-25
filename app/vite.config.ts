import { readFileSync } from 'node:fs'
import { svelte } from '@sveltejs/vite-plugin-svelte'
import { defineConfig, type Plugin } from 'vite'

const host = process.env.TAURI_DEV_HOST;

// The viewer's native FBX path needs FBXLoader's internals (its parser
// classes and module state), so vite/fbx-native.js is appended to the
// module rather than imported. Pre-bundling would skip the transform, hence
// the exclude below.
const FBX_LOADER = /three[\\/]examples[\\/]jsm[\\/]loaders[\\/]FBXLoader\.js$/
function fbxNative(): Plugin {
  const glue = new URL('./vite/fbx-native.js', import.meta.url)
  return {
    name: '3dv-fbx-native',
    transform(code, id) {
      if (!FBX_LOADER.test(id.split('?')[0])) return
      return { code: code + '\n' + readFileSync(glue, 'utf8'), map: null }
    },
  }
}

// https://v2.tauri.app/start/frontend/vite/
export default defineConfig({
  plugins: [svelte(), fbxNative()],
  optimizeDeps: { exclude: ['three/examples/jsm/loaders/FBXLoader.js'] },
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
