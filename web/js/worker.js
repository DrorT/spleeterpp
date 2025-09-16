// Web Worker: orchestrates decode -> resample -> chunk -> (inference placeholder)

import { planChunks, overlapAddStitchMono } from "./chunking.js";
import { InferenceEngine } from "./inference.js";
import { configureBackend, applyWebGLTuning } from "./tf-backend.js";

let cancelRequested = false;
let verbose = false;
let preferredBackend = "auto"; // 'auto' or 'cpu'
// Tiling settings (defaults). Can be overridden via messages or per-request.
let tilingSettings = {
  tileBatch: 3, // tiles per batched runChunks inside tiling
  prefetchDepth: 2, // future groups to precompute
  stitchWindow: "hann", // 'hann' | 'tri'
  overlapFraction: 0.5, // fraction of window shared between adjacent tiles [0, <1)
};

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
    case "query-tiling": {
      try {
        const stems = Number(payload?.stems || 2);
        const sampleRate = Number(payload?.sampleRate || 44100);
        const chunkSize = Number(payload?.chunkSize || 44100 * 10);
        const settings = {
          ...tilingSettings,
          ...(payload?.tilingSettings || {}),
        };
        self._engines = self._engines || new Map();
        const engine = self._engines.get(stems);
        let framesSpan = 512;
        try {
          const ins = (engine && engine.model && engine.model.inputs) || [];
          for (const info of ins) {
            const sh = info && info.shape;
            if (Array.isArray(sh) && sh.length === 4) {
              const tDim = sh[1];
              if (typeof tDim === "number" && tDim > 0) {
                framesSpan = tDim;
                break;
              }
            }
          }
        } catch (_) {}
        const hopSamples = 1024;
        const windowSamples = framesSpan * hopSamples;
        const overlapFrac = Math.max(
          0,
          Math.min(0.99, Number(settings.overlapFraction) || 0.5)
        );
        const hopTile = Math.max(
          1,
          Math.floor(windowSamples * (1 - overlapFrac))
        );
        const tilesForChunk =
          chunkSize <= windowSamples
            ? 1
            : Math.ceil((chunkSize - windowSamples) / hopTile) + 1;
        self.postMessage({
          type: "tiling-info",
          payload: {
            framesSpan,
            hopSamples,
            windowSamples,
            windowSeconds: windowSamples / sampleRate,
            chunkSize,
            tilesForChunk,
            overlap: overlapFrac,
            settings,
          },
        });
      } catch (err) {
        self.postMessage({
          type: "error",
          payload: makeErrorPayload(err, "query-tiling"),
        });
      }
      return;
    }
    case "set-tiling-settings": {
      try {
        if (payload && typeof payload === "object") {
          const t = Number(payload.tileBatch);
          const p = Number(payload.prefetchDepth);
          const w = String(payload.stitchWindow || tilingSettings.stitchWindow);
          const o = Number(payload.overlapFraction);
          if (Number.isFinite(t) && t >= 1 && t <= 16)
            tilingSettings.tileBatch = Math.floor(t);
          if (Number.isFinite(p) && p >= 1 && p <= 16)
            tilingSettings.prefetchDepth = Math.floor(p);
          if (w === "hann" || w === "tri") tilingSettings.stitchWindow = w;
          if (Number.isFinite(o))
            tilingSettings.overlapFraction = Math.max(0, Math.min(0.99, o));
        }
        self.postMessage({
          type: "tiling-settings-set",
          payload: { ...tilingSettings },
        });
      } catch (err) {
        self.postMessage({
          type: "error",
          payload: makeErrorPayload(err, "set-tiling-settings"),
        });
      }
      return;
    }
    case "get-tiling-settings": {
      self.postMessage({
        type: "tiling-settings",
        payload: { ...tilingSettings },
      });
      return;
    }
    case "set-backend": {
      try {
        const pref = String(payload?.preference || "auto");
        preferredBackend = [
          "auto",
          "cpu",
          "webgl",
          "webgpu",
          "wasm",
          "gpu",
        ].includes(pref)
          ? pref
          : "auto";
        const info = await configureBackend(preferredBackend);
        self.postMessage({
          type: "backend-set",
          payload: { backend: info.backend, attempts: info.attempts },
        });
      } catch (err) {
        self.postMessage({
          type: "error",
          payload: makeErrorPayload(err, "set-backend"),
        });
      }
      return;
    }
    case "set-webgl-tuning": {
      try {
        const opts = payload && typeof payload === "object" ? payload : {};
        await applyWebGLTuning(opts);
        // Re-apply backend so flags take effect if needed
        const info = await configureBackend(preferredBackend);
        self.postMessage({
          type: "webgl-tuning-set",
          payload: { ok: true, opts, backend: info.backend },
        });
      } catch (err) {
        self.postMessage({
          type: "error",
          payload: makeErrorPayload(err, "set-webgl-tuning"),
        });
      }
      return;
    }
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
        // Ensure backend is configured before model load
        try {
          await configureBackend(preferredBackend);
        } catch (_) {}
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
          batchSize = 1,
          tag: reqTag,
        } = payload;
        // Apply preferred backend before processing
        try {
          await configureBackend(preferredBackend);
        } catch (_) {}
        const tMsg0 = performance.now();
        const tTotal0 = tMsg0;
        self.postMessage({
          type: "processing-start",
          payload: { frames, chunkSize, hopSize, numStems, tag: reqTag },
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
          payload: { total: chunks.length, tag: reqTag },
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
        // Effective tiling settings for this request (allow per-request override)
        const effSettings = {
          ...tilingSettings,
          ...(payload?.tilingSettings &&
          typeof payload.tilingSettings === "object"
            ? payload.tilingSettings
            : {}),
        };
        let lastBackend = null;
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
                    stickyBackend: true,
                  });
                } catch (_) {
                  // ignore warmup errors and fall through to timed attempt
                }
              }
              const t0 = performance.now();
              const resB =
                canBatch && k > 1
                  ? await engine.runChunks(slicesB, sampleRate, {})
                  : await Promise.all(
                      slicesB.map((slice) =>
                        engine.runChunk(slice, sampleRate, {})
                      )
                    ).then((arr) => ({ stemsB: arr.map((r) => r.stems) }));
              const t1 = performance.now();
              const stemsB = resB?.stemsB || [];
              for (let j = 0; j < group.length; j++) {
                const c = group[j];
                const stems = stemsB[j];
                for (let s = 0; s < numStems; s++)
                  perStemOutputs[s].push(stems[s]);
                if (resB?.backend) lastBackend = resB.backend;
                self.postMessage({
                  type: "progress",
                  payload: {
                    tag: reqTag,
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
        // Precompute features for upcoming groups (prefetch depth)
        let nextFeaturesB = null;
        let firstBatchLogged = false;
        // Determine dynamic receptive field in samples (framesSpan * hopSamples)
        const getFramesSpan = () => {
          try {
            const ins = (engine.model && engine.model.inputs) || [];
            for (const info of ins) {
              const sh = info && info.shape;
              if (Array.isArray(sh) && sh.length === 4) {
                const tDim = sh[1];
                if (typeof tDim === "number" && tDim > 0) return tDim;
              }
            }
          } catch (_) {}
          return 512;
        };
        const FRAMES_SPAN = getFramesSpan();
        const STFT_HOP_SAMPLES = 1024; // matches inference precompute
        const MAX_WIN_SAMPLES = FRAMES_SPAN * STFT_HOP_SAMPLES;
        // Inform UI about computed receptive window and expected tiling for this chunk size
        try {
          const overlapFrac = Math.max(
            0,
            Math.min(0.99, Number(effSettings.overlapFraction) || 0.5)
          );
          const hopSamplesTile = Math.max(
            1,
            Math.floor(MAX_WIN_SAMPLES * (1 - overlapFrac))
          );
          const tilesForChunk =
            chunkSize <= MAX_WIN_SAMPLES
              ? 1
              : Math.ceil((chunkSize - MAX_WIN_SAMPLES) / hopSamplesTile) + 1;
          self.postMessage({
            type: "tiling-info",
            payload: {
              framesSpan: FRAMES_SPAN,
              hopSamples: STFT_HOP_SAMPLES,
              windowSamples: MAX_WIN_SAMPLES,
              windowSeconds: MAX_WIN_SAMPLES / sampleRate,
              chunkSize,
              tilesForChunk,
              overlap: overlapFrac,
              settings: { ...effSettings },
            },
          });
        } catch (_) {}
        // Helper: run a slice through engine, internally tiling if too long
        const runTiledMono = async (slice, sampleRate, opts) => {
          const S = (opts && opts.settings) || effSettings;
          const tag = opts?.tag;
          const tTotal0 = performance.now();
          const T = slice && slice[0] ? slice[0].length : 0;
          if (T <= 0)
            return { stems: new Array(numStems).fill(new Float32Array(0)) };
          // Dynamic receptive window length in samples
          const maxWin = MAX_WIN_SAMPLES;
          if (T <= maxWin) return engine.runChunk(slice, sampleRate, opts);
          const overlapFrac = Math.max(
            0,
            Math.min(0.99, Number(S.overlapFraction) || 0.5)
          );
          const hop = Math.max(1, Math.floor(maxWin * (1 - overlapFrac)));
          const parts = [];
          for (let s = 0; s < T; s += hop) {
            const start = s;
            const end = Math.min(T, start + maxWin);
            parts.push({ start, end });
            if (end === T) break;
          }
          // Batch the sub-tiles where possible
          let partials = [];
          let tileComputeMs = 0;
          try {
            // Batch tiles in smaller groups to reduce memory pressure
            const TILE_BATCH = Math.max(
              1,
              Math.floor(Number(S.tileBatch) || 1)
            );
            const stemsB_all = [];
            for (let i = 0; i < parts.length; i += TILE_BATCH) {
              const group = parts.slice(i, i + TILE_BATCH);
              const subSlicesB = group.map((p) =>
                slice.map((ch) => ch.subarray(p.start, p.end))
              );
              const tG0 = performance.now();
              const batched = await engine.runChunks(subSlicesB, sampleRate, {
                stickyBackend: !!opts?.stickyBackend,
                forceCpu: !!opts?.forceCpu,
                lockBackend:
                  preferredBackend === "wasm" || preferredBackend === "cpu",
              });
              const tG1 = performance.now();
              tileComputeMs += tG1 - tG0;
              const stemsB = batched && batched.stemsB ? batched.stemsB : [];
              if (!stemsB.length) throw new Error("empty batched result");
              stemsB_all.push(...stemsB);
            }
            partials = stemsB_all;
          } catch (_) {
            // Fallback to per-part singles
            partials = [];
            for (const p of parts) {
              const sub = slice.map((ch) => ch.subarray(p.start, p.end));
              const tS0 = performance.now();
              const r = await engine.runChunk(sub, sampleRate, opts);
              const tS1 = performance.now();
              tileComputeMs += tS1 - tS0;
              partials.push(r.stems);
            }
          }
          // OLA stitch for each stem with Hann crossfade
          const out = new Array(numStems)
            .fill(null)
            .map(() => new Float32Array(T));
          const weight = new Float32Array(T);
          const tri = (n) => {
            const w = new Float32Array(n);
            if (n === 1) {
              w[0] = 1;
              return w;
            }
            for (let i = 0; i < n; i++)
              w[i] = 1 - Math.abs((2 * i) / (n - 1) - 1);
            return w;
          };
          const hann = (n) => {
            const w = new Float32Array(n);
            if (n === 1) {
              w[0] = 1;
              return w;
            }
            for (let i = 0; i < n; i++)
              w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
            return w;
          };
          const makeWindow = (n) =>
            S.stitchWindow === "tri" ? tri(n) : hann(n);
          const tStitch0 = performance.now();
          for (let pi = 0; pi < parts.length; pi++) {
            const p = parts[pi];
            const len = p.end - p.start;
            const wv = makeWindow(len);
            for (let s = 0; s < numStems; s++) {
              const src = partials[pi][s];
              for (let t = 0; t < len; t++) {
                const w = wv[t];
                out[s][p.start + t] += src[t] * w;
                if (s === 0) weight[p.start + t] += w; // weight once
              }
            }
          }
          // normalize
          for (let t = 0; t < T; t++) {
            const w = weight[t] > 1e-6 ? 1 / weight[t] : 1;
            for (let s = 0; s < numStems; s++) out[s][t] *= w;
          }
          const tStitch1 = performance.now();
          const tTotal1 = performance.now();
          if (verbose) {
            self.postMessage({
              type: "tiling-timing",
              payload: {
                tag,
                samples: T,
                tiles: parts.length,
                tileBatch: Math.max(1, Math.floor(Number(S.tileBatch) || 1)),
                stitchWindow: S.stitchWindow,
                overlapFraction: Number(S.overlapFraction) || 0.5,
                tileComputeMs: tileComputeMs,
                stitchMs: tStitch1 - tStitch0,
                totalMs: tTotal1 - tTotal0,
              },
            });
          }
          return { stems: out };
        };

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
              Math.min(
                chunks.length,
                i +
                  group.length +
                  bs *
                    Math.max(
                      1,
                      Math.floor(Number(effSettings.prefetchDepth) || 1)
                    )
              )
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
              // If any slice exceeds dynamic max window, handle per-slice with internal tiling
              const tooLong = slicesB.some(
                (sl) => sl[0].length > MAX_WIN_SAMPLES
              );
              if (tooLong) {
                const stemsB = [];
                for (let j = 0; j < group.length; j++) {
                  const r = await runTiledMono(slicesB[j], sampleRate, {
                    stickyBackend: true,
                    forceCpu: preferredBackend === "cpu",
                    settings: effSettings,
                    tag: reqTag,
                  });
                  stemsB.push(r.stems);
                }
                resB = { stemsB };
              } else {
                resB = await engine.runChunks(slicesB, sampleRate, {
                  stickyBackend: true,
                  forceCpu: preferredBackend === "cpu",
                  lockBackend:
                    preferredBackend === "wasm" || preferredBackend === "cpu",
                  featuresB:
                    nextFeaturesB && nextFeaturesB.length === slicesB.length
                      ? nextFeaturesB
                      : undefined,
                });
              }
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
                  stickyBackend: true,
                  forceCpu: preferredBackend === "cpu",
                  lockBackend:
                    preferredBackend === "wasm" || preferredBackend === "cpu",
                });
                const tS1 = performance.now();
                for (let s = 0; s < numStems; s++)
                  perStemOutputs[s].push(res.stems[s]);
                if (res?.backend) lastBackend = res.backend;
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
            if (verbose)
              self.postMessage({
                type: "debug",
                payload: {
                  message: `[worker] batched group time ${(t1 - t0).toFixed(
                    1
                  )} ms`,
                },
              });
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
                    tag: reqTag,
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
            const result = await runTiledMono(slice, sampleRate, {
              stickyBackend: true,
              forceCpu: preferredBackend === "cpu",
              lockBackend:
                preferredBackend === "wasm" || preferredBackend === "cpu",
              features: nextFeaturesB || (await nextFeatPromise),
              settings: effSettings,
              tag: reqTag,
            });
            const t1 = performance.now();
            nextFeaturesB = null;
            for (let s = 0; s < numStems; s++)
              perStemOutputs[s].push(result.stems[s]);
            if (result?.backend) lastBackend = result.backend;
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
                  tag: reqTag,
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
        // Reconstruction metric: how close is sum(stems) to original mono mix
        const computeRecon = (stems, channels) => {
          try {
            const n = Math.min(
              channels && channels.length ? channels[0].length : 0,
              stems && stems.length ? stems[0].length : 0
            );
            if (!n) return null;
            const stride = n > 200000 ? Math.ceil(n / 200000) : 1;
            let sumSq = 0;
            let sumSqMix = 0;
            let count = 0;
            const C = Math.max(1, channels.length || 0);
            for (let i = 0; i < n; i += stride) {
              let mix = 0;
              for (let c = 0; c < C; c++) mix += channels[c][i] || 0;
              mix /= C;
              let sumS = 0;
              for (let s = 0; s < stems.length; s++) sumS += stems[s][i] || 0;
              const d = sumS - mix;
              sumSq += d * d;
              sumSqMix += mix * mix;
              count++;
            }
            const rmsDiff = Math.sqrt(sumSq / Math.max(1, count));
            const rmsMix = Math.sqrt(sumSqMix / Math.max(1, count));
            const relPct = rmsMix > 1e-12 ? (rmsDiff / rmsMix) * 100 : 0;
            return { rmsDiff, rmsMix, relPct, samples: count };
          } catch (_) {
            return null;
          }
        };
        const recon = computeRecon(stitched, channels);
        const tDone0 = performance.now();
        const transfer = stitched.map((a) => a.buffer);
        self.postMessage(
          {
            type: "done",
            payload: {
              tag: reqTag,
              stems: stitched,
              sampleRate,
              stats,
              recon,
              backend: lastBackend,
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
        const ep = makeErrorPayload(err, "process-audio");
        self.postMessage({
          type: "error",
          payload: { ...ep, tag: payload?.tag },
        });
      }
      return;
    }
  }
};
