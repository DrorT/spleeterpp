// Web Worker: orchestrates decode -> resample -> chunk -> (inference placeholder)

import { planChunks, overlapAddStitchMono } from "./chunking.js";
import { InferenceEngine } from "./inference.js";

let cancelRequested = false;

// Signal readiness as soon as the module loads
self.postMessage({ type: "worker-ready" });

self.onmessage = async (e) => {
  const { type, payload } = e.data || {};
  self.postMessage({
    type: "debug",
    payload: { message: `Worker received message: ${type}` },
  });
  switch (type) {
    case "load-model": {
      try {
        const stems = Number(payload?.stems || 2);
        self.postMessage({ type: "model-loading", payload: { stems } });
        const engine = new InferenceEngine({ numStems: stems });
        const info = await engine.load(stems);
        // cache current engine by stems count if needed later
        self._engines = self._engines || new Map();
        self._engines.set(stems, engine);
        const ioDetail = {
          inputs: (engine.model?.inputs || []).map((t) => ({
            name: t?.name,
            dtype: t?.dtype,
            shape: t?.shape,
          })),
          outputs: (engine.model?.outputs || []).map((t) => ({
            name: t?.name,
            dtype: t?.dtype,
            shape: t?.shape,
          })),
        };
        self.postMessage({
          type: "model-loaded",
          payload: { stems, info, ioDetail },
        });
      } catch (err) {
        self.postMessage({
          type: "error",
          payload: {
            message:
              "Model load failed: " + ((err && err.message) || String(err)),
          },
        });
      }
      return;
    }
    case "preload-all-models": {
      try {
        self.postMessage({ type: "preload-start" });
        const stemsList = payload?.stemsList || [2, 4, 5];
        self._engines = self._engines || new Map();
        const results = [];
        for (let i = 0; i < stemsList.length; i++) {
          const stems = stemsList[i];
          let ok = true;
          let info = null;
          try {
            const engine = new InferenceEngine({ numStems: stems });
            info = await engine.load(stems);
            self._engines.set(stems, engine);
          } catch (e) {
            ok = false;
          }
          results.push({ stems, ok });
          self.postMessage({
            type: "model-progress",
            payload: { index: i + 1, total: stemsList.length, stems, ok },
          });
        }
        self.postMessage({ type: "models-preloaded", payload: { results } });
      } catch (err) {
        self.postMessage({
          type: "error",
          payload: {
            message: "Preload failed: " + ((err && err.message) || String(err)),
          },
        });
      }
      return;
    }
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
        self.postMessage({
          type: "processing-start",
          payload: { frames, chunkSize, hopSize, numStems },
        });
        const chunks = planChunks(frames, chunkSize, hopSize);
        self.postMessage({
          type: "planned",
          payload: { total: chunks.length },
        });
        // Use cached engine if available or create fresh
        self._engines = self._engines || new Map();
        let engine = self._engines.get(numStems);
        if (!engine) {
          engine = new InferenceEngine({ numStems });
          await engine.load(numStems);
          self._engines.set(numStems, engine);
        }
        const perStemOutputs = new Array(numStems).fill(null).map(() => []);
        let errorOnce = false;
        for (let i = 0; i < chunks.length; i++) {
          if (cancelRequested) {
            self.postMessage({ type: "cancelled" });
            return;
          }
          const c = chunks[i];
          // Slice channels per chunk
          const slice = channels.map((ch) => ch.subarray(c.start, c.end));
          let result;
          let t0, t1;
          try {
            t0 = performance.now();
            result = await engine.runChunk(slice, sampleRate);
            t1 = performance.now();
          } catch (e) {
            if (!errorOnce) {
              self.postMessage({
                type: "debug",
                payload: {
                  message: `[worker] runChunk failed: ${
                    e && e.message ? e.message : e
                  }`,
                },
              });
              errorOnce = true;
            }
            throw e;
          }
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
              durationMs: t1 - t0,
            },
          });
        }
        // Stitch per-stem results (mono only for now)
        const stitched = perStemOutputs.map((chunks) =>
          overlapAddStitchMono(chunks, frames, chunkSize, hopSize)
        );
        const computeStats = (arr) => {
          const n = arr.length;
          const maxSamples = 200000;
          const stride = n > maxSamples ? Math.ceil(n / maxSamples) : 1;
          let min = Infinity,
            max = -Infinity,
            sumSq = 0,
            count = 0,
            nz = 0;
          for (let i = 0; i < n; i += stride) {
            const v = arr[i];
            if (v < min) min = v;
            if (v > max) max = v;
            sumSq += v * v;
            if (v !== 0) nz++;
            count++;
          }
          const rms = Math.sqrt(sumSq / Math.max(1, count));
          const nzFrac = count ? nz / count : 0;
          return { len: n, min, max, rms, nzFrac };
        };
        const stats = stitched.map((a) => computeStats(a));
        self.postMessage({
          type: "done",
          payload: { stems: stitched, sampleRate, stats },
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
