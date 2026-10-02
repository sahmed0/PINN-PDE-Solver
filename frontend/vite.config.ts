import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // plotly.js's module sources (see src/lib/plotlyCore.ts) read Node's `global`, which the prebuilt
  // distribution used to shim. Map it to the browser's globalThis in both dev and build.
  define: { global: 'globalThis' },
})
