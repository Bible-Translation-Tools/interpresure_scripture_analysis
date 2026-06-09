import { defineConfig } from 'vite'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  base: "/interpresure_scripture_analysis/",
  server: {
    port: 3006,
  },
  plugins: [
    tailwindcss(),
    react()
],
})
