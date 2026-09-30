import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API is a separate process on 8000. Proxying rather than calling it
// cross-origin means the app is same-origin in development and in production,
// so there is one code path and no CORS surprises at the end.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: { outDir: "dist" },
});
