import vinext from "vinext";
import { defineConfig } from "vite";

const backend = process.env.ROC_CALIB_API ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [vinext()],
  server: {
    host: "127.0.0.1",
    proxy: {
      "/api/segment/image": { target: backend, rewrite: () => "/v1/image/segment" },
      "/api/segment/health": { target: backend, rewrite: () => "/health" },
      "/api/calibration": { target: backend, rewrite: (path: string) => path.replace(/^\/api\/calibration/, "/v1/calibration") },
      "/data/cache": { target: backend, rewrite: (path: string) => path.replace(/^\/data\/cache/, "/v1/prepared") },
    },
  },
});
