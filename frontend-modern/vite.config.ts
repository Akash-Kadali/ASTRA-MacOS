import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "/app/",
  build: {
    outDir: "../frontend/app",
    emptyOutDir: true,
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          "react-runtime": ["react", "react-dom"],
          "query-runtime": ["@tanstack/react-query"],
          "ui-runtime": ["@radix-ui/react-dialog", "motion/react", "lucide-react"],
        },
      },
    },
  },
  server: { proxy: { "/api": "http://127.0.0.1:8000", "/health": "http://127.0.0.1:8000" } }
});
