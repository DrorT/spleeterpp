// Web Worker: orchestrates decode -> resample -> chunk -> (inference placeholder)

import { planChunks, overlapAddStitchMono } from "./chunking.js";
import { InferenceEngine } from "./inference.js";

let cancelRequested = false;
let verbose = false;

// Signal readiness as soon as the module loads
self.postMessage({ type: "worker-ready" });

function makeErrorPayload(err, context) {
  const message = (err && err.message) || String(err);
  const stack = (err && err.stack) || "";
  let site = "";
  try {
    const lines = String(stack).split("\n");
    for (const ln of lines) {
      const m = ln.match(/(\(?)([^)\s]+):(\d+):(\d+)\)?$/);
      if (m) {
        site = `${m[2]}:${m[3]}:${m[4]}`;
        break;
      }
    }
  } catch (_) {}
  return { context, message, site, stack, name: err?.name };
}

self.onmessage = async (e) => {
  const { type, payload } = e.data || {};
  // Keep initial debug minimal; verbose logs are filtered in UI
  // self.postMessage({ type: "debug", payload: { message: `Worker received message: ${type}` } });
  switch (type) {
    case "set-verbose": {
      verbose = !!payload?.verbose;
      self.postMessage({
        type: "debug",
        payload: { message: `[worker] verbose=${verbose}` },
      });
      return;
    }
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
          payload: makeErrorPayload(err, "load-model"),
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
          payload: makeErrorPayload(err, "preload-all-models"),
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
          precision = "auto",
          batchSize = 1,
        } = payload;
        const tMsg0 = performance.now();
        const tTotal0 = tMsg0;
        self.postMessage({
          type: "processing-start",
          payload: { frames, chunkSize, hopSize, numStems },
        });
        const tMsg1 = performance.now();
        if (verbose)
          self.postMessage({
            type: "debug",
            payload: {
              message: `[worker] post processing-start took ${(
                tMsg1 - tMsg0
              ).toFixed(2)}ms`,
            },
          });
        const chunks = planChunks(frames, chunkSize, hopSize);
        const tPlan0 = performance.now();
        self.postMessage({
          type: "planned",
          payload: { total: chunks.length },
        });
        const tPlan1 = performance.now();
        if (verbose)
          self.postMessage({
            type: "debug",
            payload: {
              message: `[worker] post planned took ${(tPlan1 - tPlan0).toFixed(
                2
              )}ms`,
            },
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
        const progressIntervalMs = Math.max(
          50,
          Number(payload?.progressIntervalMs || 250)
        );
        let lastProgressTime = 0;
        let errorOnce = false;
        const isAuto = String(batchSize) === "auto";
        const bsFixed = Math.max(1, Math.min(4, Number(batchSize) || 1));
        let bs = bsFixed;
        const canBatch =
          typeof engine.supportsBatching === "function" &&
          !!engine.supportsBatching();
        if (verbose)
          self.postMessage({
            type: "debug",
            payload: {
              message: `[worker] batching requested=${String(
                batchSize
              )} supported=${canBatch}`,
            },
          });
        // Declare loop index before any closures that may reference it
        let i = 0;
        const chooseAuto = async () => {
          const tried = [];
          let idx = 0;
          const maxProbe = Math.min(chunks.length - idx, 4);
          for (let k = 1; k <= 4 && k <= maxProbe; k++) {
            const group = chunks.slice(idx, idx + k);
            const slicesB = group.map((c) =>
              channels.map((ch) => ch.subarray(c.start, c.end))
            );
            try {
              // Pre-warm for batched shapes so timing is fair
              if (canBatch && k > 1) {
                try {
                  await engine.runChunks(slicesB, sampleRate, {
                    precision,
                    stickyBackend: true,
                  });
                } catch (_) {
                  // ignore warmup errors and fall through to timed attempt
                }
              }
              const t0 = performance.now();
              const resB =
                canBatch && k > 1
                  ? await engine.runChunks(slicesB, sampleRate, { precision })
                  : await Promise.all(
                      slicesB.map((slice) =>
                        engine.runChunk(slice, sampleRate, { precision })
                      )
                    ).then((arr) => ({ stemsB: arr.map((r) => r.stems) }));
              const t1 = performance.now();
              const stemsB = resB?.stemsB || [];
              for (let j = 0; j < group.length; j++) {
                const c = group[j];
                const stems = stemsB[j];
                for (let s = 0; s < numStems; s++)
                  perStemOutputs[s].push(stems[s]);
                self.postMessage({
                  type: "progress",
                  payload: {
                    index: idx + j + 1,
                    total: chunks.length,
                    start: c.start,
                    end: c.end,
                    durationMs: (t1 - t0) / group.length,
                  },
                });
              }
              tried.push({ k, msPerChunk: (t1 - t0) / group.length, ok: true });
              idx += group.length;
            } catch (_) {
              tried.push({ k, msPerChunk: Infinity, ok: false });
              break;
            }
          }
          if (tried.length === 0) return 1;
          const ok = tried.filter((t) => t.ok);
          if (ok.length === 0) return 1;
          ok.sort((a, b) => a.msPerChunk - b.msPerChunk);
          const best = ok[0];
          self.postMessage({
            type: "batch-auto-selected",
            payload: { k: best.k, results: tried },
          });
          // Advance i by processed chunks
          i = idx;
          return best.k;
        };
        if (isAuto) bs = await chooseAuto();
        // Prepare a simple feature precompute pipeline
        const precompute = (slice) => {
          try {
            if (typeof engine.precomputeFeatures === "function") {
              return engine.precomputeFeatures(slice);
            }
          } catch (_) {}
          return null;
        };
        // Precompute features for first group/slice
        let nextFeaturesB = null;
        let firstBatchLogged = false;
        while (i < chunks.length) {
          if (cancelRequested) {
            self.postMessage({ type: "cancelled" });
            return;
          }
          if (canBatch && bs > 1) {
            const group = chunks.slice(i, Math.min(chunks.length, i + bs));
            const slicesB = group.map((c) =>
              channels.map((ch) => ch.subarray(c.start, c.end))
            );
            // Kick off precompute for the next group early
            const nextGroup = chunks.slice(
              i + group.length,
              Math.min(chunks.length, i + group.length + bs)
            );
            const nextSlicesB = nextGroup.map((c) =>
              channels.map((ch) => ch.subarray(c.start, c.end))
            );
            const nextFeatPromise =
              nextSlicesB.length > 0
                ? Promise.all(nextSlicesB.map((sl) => precompute(sl)))
                : null;
            let resB;
            let t0 = performance.now();
            try {
              resB = await engine.runChunks(slicesB, sampleRate, {
                precision,
                stickyBackend: true,
                featuresB:
                  nextFeaturesB && nextFeaturesB.length === slicesB.length
                    ? nextFeaturesB
                    : undefined,
              });
              if (!firstBatchLogged && verbose) {
                firstBatchLogged = true;
                self.postMessage({
                  type: "debug",
                  payload: {
                    message:
                      "First batched group may include JIT/warmup cost; later groups should be faster.",
                  },
                });
              }
            } catch (e) {
              // Fallback to single for each in group
              for (let j = 0; j < group.length; j++) {
                const c = group[j];
                const slice = slicesB[j];
                const tS0 = performance.now();
                const res = await engine.runChunk(slice, sampleRate, {
                  precision,
                  stickyBackend: true,
                });
                const tS1 = performance.now();
                for (let s = 0; s < numStems; s++)
                  perStemOutputs[s].push(res.stems[s]);
                self.postMessage({
                  type: "progress",
                  payload: {
                    index: i + j + 1,
                    total: chunks.length,
                    start: c.start,
                    end: c.end,
                    durationMs: tS1 - tS0,
                  },
                });
              }
              i += group.length;
              continue;
            }
            const t1 = performance.now();
            // Resolve next features after execute completes
            nextFeaturesB = nextFeatPromise ? await nextFeatPromise : null;
            const stemsB = resB?.stemsB || [];
            for (let j = 0; j < group.length; j++) {
              const c = group[j];
              const stems = stemsB[j];
              for (let s = 0; s < numStems; s++)
                perStemOutputs[s].push(stems[s]);
              const now = performance.now();
              if (
                now - lastProgressTime >= progressIntervalMs ||
                i + j === chunks.length - 1
              ) {
                const tProg0 = now;
                self.postMessage({
                  type: "progress",
                  payload: {
                    index: i + j + 1,
                    total: chunks.length,
                    start: c.start,
                    end: c.end,
                    durationMs: (t1 - t0) / group.length,
                  },
                });
                const tProg1 = performance.now();
                if (verbose)
                  self.postMessage({
                    type: "debug",
                    payload: {
                      message: `[worker] progress ${i + j + 1}/${
                        chunks.length
                      } batched`,
                    },
                  });
                lastProgressTime = now;
              }
            }
            i += group.length;
          } else {
            const c = chunks[i];
            const slice = channels.map((ch) => ch.subarray(c.start, c.end));
            // Start computing features for next slice early
            const nextC = chunks[i + 1];
            const nextSlice =
              nextC &&
              channels.map((ch) => ch.subarray(nextC.start, nextC.end));
            const nextFeatPromise = nextSlice ? precompute(nextSlice) : null;
            let t0 = performance.now();
            const result = await engine.runChunk(slice, sampleRate, {
              precision,
              stickyBackend: true,
              features: nextFeaturesB || (await nextFeatPromise),
            });
            const t1 = performance.now();
            nextFeaturesB = null;
            for (let s = 0; s < numStems; s++)
              perStemOutputs[s].push(result.stems[s]);
            await new Promise((r) => setTimeout(r, 0));
            const now = performance.now();
            if (
              now - lastProgressTime >= progressIntervalMs ||
              i === chunks.length - 1
            ) {
              const tProg0 = now;
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
              const tProg1 = performance.now();
              if (verbose)
                self.postMessage({
                  type: "debug",
                  payload: {
                    message: `[worker] post progress ${i + 1}/${
                      chunks.length
                    } took ${(tProg1 - tProg0).toFixed(2)}ms`,
                  },
                });
              lastProgressTime = now;
            }
            i += 1;
          }
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
        const tDone0 = performance.now();
        const transfer = stitched.map((a) => a.buffer);
        self.postMessage(
          {
            type: "done",
            payload: {
              stems: stitched,
              sampleRate,
              stats,
              totalMs: tDone0 - tTotal0,
            },
          },
          transfer
        );
        const tDone1 = performance.now();
        if (verbose)
          self.postMessage({
            type: "debug",
            payload: {
              message: `[worker] post done took ${(tDone1 - tDone0).toFixed(
                2
              )}ms`,
            },
          });
      } catch (err) {
        self.postMessage({
          type: "error",
          payload: makeErrorPayload(err, "process-audio"),
        });
      }
      return;
    }
  }
};
