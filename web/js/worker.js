// Web Worker: orchestrates decode -> resample -> chunk -> (inference placeholder)

import { planChunks, overlapAddStitchMono } from "./chunking.js";
import { InferenceEngine } from "./inference.js";

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
        const {
          sampleRate,
          frames,
          chunkSize,
          hopSize,
          channels,
          numStems = 2,
        } = payload;
        const chunks = planChunks(frames, chunkSize, hopSize);
        self.postMessage({
          type: "planned",
          payload: { total: chunks.length },
        });
        const engine = new InferenceEngine({ numStems });
        // engine.load(modelUrl) // TODO when model loader is ready
        const perStemOutputs = new Array(numStems).fill(null).map(() => []);
        for (let i = 0; i < chunks.length; i++) {
          if (cancelRequested) {
            self.postMessage({ type: "cancelled" });
            return;
          }
          const c = chunks[i];
          // Slice mono channel for now (first channel)
          const mono = channels[0].subarray(c.start, c.end);
          // Placeholder inference: pass-through; later call await engine.runChunk(mono, sampleRate)
          const result = {
            stems: new Array(numStems).fill(null).map(() => mono),
          };
          for (let s = 0; s < numStems; s++)
            perStemOutputs[s].push(result.stems[s]);
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
        // Stitch per-stem results (mono only for now)
        const stitched = perStemOutputs.map((chunks) =>
          overlapAddStitchMono(chunks, frames, chunkSize, hopSize)
        );
        self.postMessage({
          type: "done",
          payload: { stems: stitched, sampleRate },
        });
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
