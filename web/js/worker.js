// Web Worker: orchestrates decode -> resample -> chunk -> (inference placeholder)

import { planChunks } from "./chunking.js";

let cancelRequested = false;

self.onmessage = async (e) => {
  const { type, payload } = e.data || {};
  switch (type) {
    case "cancel":
      cancelRequested = true;
      return;
    case "process-audio": {
      cancelRequested = false;
      try {
        const { sampleRate, frames, chunkSize, hopSize } = payload;
        const chunks = planChunks(frames, chunkSize, hopSize);
        self.postMessage({
          type: "planned",
          payload: { total: chunks.length },
        });
        for (let i = 0; i < chunks.length; i++) {
          if (cancelRequested) {
            self.postMessage({ type: "cancelled" });
            return;
          }
          const c = chunks[i];
          // Placeholder for inference work per chunk
          await new Promise((r) => setTimeout(r, 0));
          self.postMessage({
            type: "progress",
            payload: {
              index: i + 1,
              total: chunks.length,
              start: c.start,
              end: c.end,
            },
          });
        }
        self.postMessage({ type: "done", payload: {} });
      } catch (err) {
        self.postMessage({
          type: "error",
          payload: { message: (err && err.message) || String(err) },
        });
      }
      return;
    }
  }
};
