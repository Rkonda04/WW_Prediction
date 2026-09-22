import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Relative base so the built bundle can be dropped into any sub-path of the
  // City portal without rewriting asset URLs.
  base: './',
  build: { outDir: 'dist', sourcemap: false },
})
