// Inference adapter for TF.js models (model loading + inference)
import { ensureTF, configureBackend } from "./tf-backend.js";
import { loadGraphModelWithCache } from "./model-loader.js";
import { stftStereo } from "./fft4096.js";

export class InferenceEngine {
  constructor(opts = {}) {
    this.numStems = opts.numStems || 2;
    this.model = null;
    this.modelUrl = null;
    this.io = { inputs: [], outputs: [] };
    this._fillNonAudioWithNoise = true;
    this._zeroPadCache = new Map(); // key: `${type}:${len}:${bins}:${ch}` -> tf.Tensor
  }

  // Returns true if the current model IO indicates a batch dimension that we can exploit
  // without changing time-frequency patch shapes. Conservative default: true if any input
  // has rank >= 2 with a first dim unspecified/negative.
  supportsBatching() {
    const inputs = this.model?.inputs || [];
    for (const i of inputs) {
      const shape = i?.shape || [];
      if (!Array.isArray(shape) || shape.length < 2) continue;
      const b0 = shape[0];
      if (b0 == null || (typeof b0 === "number" && b0 < 0)) return true;
    }
    return false;
  }

  _stemOutputNames(n) {
    if (n === 2) return ["strided_slice_13", "strided_slice_23"];
    if (n === 4)
      return [
        "strided_slice_13",
        "strided_slice_23",
        "strided_slice_33",
        "strided_slice_43",
      ];
    if (n === 5)
      return [
        "strided_slice_18",
        "strided_slice_38",
        "strided_slice_48",
        "strided_slice_28",
        "strided_slice_58",
      ];
    return [];
  }

  async load(modelUrlOrStems) {
    const tf = await ensureTF();
    // Prefer GPU (WebGL) if available; fall back to CPU on failure
    try {
      await configureBackend("auto"); // tries webgl, then wasm, then cpu
    } catch (_) {
      try {
        await configureBackend("cpu");
      } catch (_) {}
    }
    const modelUrl =
      typeof modelUrlOrStems === "number"
        ? `/models/${modelUrlOrStems}stems/model.json`
        : modelUrlOrStems;
    if (typeof modelUrlOrStems === "number") this.numStems = modelUrlOrStems;
    this.modelUrl = modelUrl;
    this.model = await loadGraphModelWithCache(modelUrl);
    try {
      const inputs = (this.model.inputs || []).map((i) => (i && i.name) || "");
      const outputs = (this.model.outputs || []).map(
        (o) => (o && o.name) || ""
      );
      this.io = { inputs, outputs };
    } catch (_) {
      this.io = { inputs: [], outputs: [] };
    }
    return { url: this.modelUrl, backend: tf.getBackend(), io: this.io };
  }

  isLoaded() {
    return !!this.model;
  }

  info() {
    return { url: this.modelUrl, io: this.io, numStems: this.numStems };
  }

  // Optional offline feature precompute for pipelining
  // Returns { time, outBins, tSpan, stftReal, stftImag, magArr }
  // This is CPU-only and independent of tf backend.
  precomputeFeatures(channelsOrMono) {
    const channels = Array.isArray(channelsOrMono)
      ? channelsOrMono
      : [channelsOrMono];
    const { channels: st, frames, bins } = stftStereo(channels, 1024, 512);
    let time = frames;
    if (!time || time <= 0) time = 1;
    const outBins = 2049;
    const ch0 = st[0];
    const ch1 = st.length > 1 ? st[1] : st[0];
    const realArr = new Float32Array(time * outBins * 2);
    const imagArr = new Float32Array(time * outBins * 2);
    for (let f = 0; f < time; f++) {
      const base = f * outBins;
      for (let k = 0; k < outBins; k++) {
        const i0 = base + k;
        const dst = (base + k) * 2;
        realArr[dst + 0] = ch0?.real?.[i0] ?? 0;
        realArr[dst + 1] = ch1?.real?.[i0] ?? 0;
        imagArr[dst + 0] = ch0?.imag?.[i0] ?? 0;
        imagArr[dst + 1] = ch1?.imag?.[i0] ?? 0;
      }
    }
    const magArr = new Float32Array(time * 1024 * 2);
    for (let f = 0; f < time; f++) {
      const base = f * outBins;
      const dstBase = f * 1024 * 2;
      for (let k = 0; k < 1024; k++) {
        const i0 = base + k;
        const r0 = ch0?.real?.[i0] ?? 0;
        const i0v = ch0?.imag?.[i0] ?? 0;
        const r1 = ch1?.real?.[i0] ?? 0;
        const i1v = ch1?.imag?.[i0] ?? 0;
        magArr[dstBase + k * 2 + 0] = Math.hypot(r0, i0v);
        magArr[dstBase + k * 2 + 1] = Math.hypot(r1, i1v);
      }
    }
    const tSpan = Math.min(512, time);
    return {
      time,
      outBins,
      tSpan,
      stftReal: realArr,
      stftImag: imagArr,
      magArr,
    };
  }

  _getZeroPad(len, bins, ch, type = "mag") {
    // type: "mag" (shape [len, 1024, 2]) or "complex" (shape [len, 2049, 2])
    // Returns a cached tf.zeros tensor of requested shape.
    const key = `${type}:${len}:${bins}:${ch}`;
    let t = this._zeroPadCache.get(key);
    if (t && !t.isDisposedInternal) return t;
    // Lazily create and cache
    const tf =
      typeof window !== "undefined" && window.tf
        ? window.tf
        : typeof self !== "undefined" && self.tf
        ? self.tf
        : null;
    if (!tf) throw new Error("TF.js not initialized for zero pad");
    t = tf.zeros([len, bins, ch], "float32");
    this._zeroPadCache.set(key, t);
    return t;
  }

  async runChunk(channelsOrMono, sampleRate, opts = {}) {
    // Yield to the event loop before the heavy execution part
    await new Promise((resolve) => setTimeout(resolve, 0));
    return this._runChunkInternal(channelsOrMono, sampleRate, opts);
  }

  // Batched version: channelsB is an array length B of channel arrays
  // channelsB[b] = [ch0Float32Array, ch1Float32Array, ...]
  async runChunks(channelsB, sampleRate, opts = {}) {
    const tf = await ensureTF();
    if (!Array.isArray(channelsB) || channelsB.length === 0)
      return { stemsB: [] };
    // Fallback to single if model missing
    if (!this.model) {
      const single = await this.runChunk(channelsB[0], sampleRate, opts);
      this._lastExecBackend = single.backend || this._lastExecBackend;
      return { stemsB: [single.stems], backend: this._lastExecBackend };
    }
    const B = channelsB.length;
    // Probe input layout from first sample using runChunk logic up to tensor creation
    const first = channelsB[0];
    const inputsInfo = this.model.inputs || [];
    const getRank = (i) => (i && i.shape ? i.shape.length : 0);
    const f32Inputs = inputsInfo.filter((i) => i && i.dtype === "float32");
    let audioInputInfo = null;
    const rank2or3 = f32Inputs.filter((i) => {
      const r = getRank(i);
      return r === 2 || r === 3;
    });
    if (rank2or3.length) audioInputInfo = rank2or3[0];
    else if (f32Inputs.length)
      audioInputInfo = f32Inputs.sort((a, b) => getRank(a) - getRank(b))[0];
    else audioInputInfo = inputsInfo[0] || null;
    const inShape = audioInputInfo?.shape ? audioInputInfo.shape.slice() : null;
    const rank = inShape ? inShape.length : 3;
    const dims = (inShape || []).map((d) => (d == null || d < 0 ? null : d));
    const prepOne = (channels) => {
      const Cexp = Math.max(1, channels.length);
      let T = channels[0].length;
      const prepped = channels.map((ch) => ch);
      let layout = "BTC";
      if (rank === 2) {
        const d0 = dims[0];
        const d1 = dims[1];
        if (d1 === channels.length || d1 === 1 || d1 == null) layout = "TC";
        else if (d0 === channels.length || d0 === 1) layout = "CT";
        else layout = "TC";
      } else if (rank >= 3) {
        const d1 = dims[1];
        const d2 = dims[2];
        if (d2 === channels.length || d2 === 1 || d2 == null) layout = "BTC";
        else if (d1 === channels.length || d1 === 1) layout = "BCT";
        else layout = "BTC";
      }
      return { layout, T, C: channels.length, prepped };
    };
    const firstPrep = prepOne(first);
    const makeAudioTensor = (channels) => {
      const { layout, T, C, prepped } = prepOne(channels);
      if (layout === "TC") {
        const inter = new Float32Array(T * C);
        let idx = 0;
        for (let t = 0; t < T; t++)
          for (let c = 0; c < C; c++) inter[idx++] = prepped[c][t];
        return { t: tf.tensor(inter, [1, T, C], "float32"), layout: "BTC" };
      }
      if (layout === "CT") {
        const byCh = new Float32Array(T * C);
        let idx = 0;
        for (let c = 0; c < C; c++)
          for (let t = 0; t < T; t++) byCh[idx++] = prepped[c][t];
        return { t: tf.tensor(byCh, [1, C, T], "float32"), layout: "BCT" };
      }
      if (layout === "BCT") {
        const byCh = new Float32Array(T * C);
        let idx = 0;
        for (let c = 0; c < C; c++)
          for (let t = 0; t < T; t++) byCh[idx++] = prepped[c][t];
        return { t: tf.tensor(byCh, [1, C, T], "float32"), layout: "BCT" };
      }
      // BTC
      const inter = new Float32Array(T * C);
      let idx = 0;
      for (let t = 0; t < T; t++)
        for (let c = 0; c < C; c++) inter[idx++] = prepped[c][t];
      return { t: tf.tensor(inter, [1, T, C], "float32"), layout: "BTC" };
    };
    const audioTensors = channelsB.map(makeAudioTensor);
    const layout = audioTensors[0].layout;
    const stacked = tf.concat(
      audioTensors.map((x) => x.t),
      0
    ); // [B,*,*]
    audioTensors.forEach((x) => x.t.dispose());
    const tensorsToDispose = [stacked];
    const inputsInfo2 = this.model.inputs || [];
    const minimalInputs = {};
    if (audioInputInfo?.name) minimalInputs[audioInputInfo.name] = stacked;
    const matchesComplexInput = (info) =>
      info &&
      info.dtype === "complex64" &&
      Array.isArray(info.shape) &&
      info.shape.length === 3 &&
      info.shape[2] === 2 &&
      info.shape[1] === 2049;
    const matchesMag4dInput = (info) =>
      info &&
      info.dtype === "float32" &&
      Array.isArray(info.shape) &&
      info.shape.length === 4 &&
      info.shape[1] === 512 &&
      info.shape[2] === 1024 &&
      info.shape[3] === 2;
    // Build feature batches if required
    const haveComplex = inputsInfo2.some(matchesComplexInput);
    const haveMag = inputsInfo2.some(matchesMag4dInput);
    try {
      if (haveComplex || haveMag) {
        // Use precomputed features if provided; else compute now
        const feats =
          opts &&
          Array.isArray(opts.featuresB) &&
          opts.featuresB.length === channelsB.length
            ? opts.featuresB
            : channelsB.map((ch) => this.precomputeFeatures(ch));
        if (haveComplex) {
          const comps = feats.map((f) => {
            const realT = tf.tensor(f.stftReal, [f.time, 2049, 2], "float32");
            const imagT = tf.tensor(f.stftImag, [f.time, 2049, 2], "float32");
            const rs = realT.slice([0, 0, 0], [f.tSpan, 2049, 2]);
            const is = imagT.slice([0, 0, 0], [f.tSpan, 2049, 2]);
            const needPad = 512 - f.tSpan;
            const rpad =
              needPad > 0
                ? this._getZeroPad(needPad, 2049, 2, "complex")
                : null;
            const ipad = needPad > 0 ? rpad : null; // reuse same zeros
            const r512 = rpad ? tf.concat([rs, rpad], 0) : rs;
            const i512 = ipad ? tf.concat([is, ipad], 0) : is;
            realT.dispose();
            imagT.dispose();
            rs.dispose();
            is.dispose();
            // cached pads are persistent; do not dispose
            const c = tf.complex(r512, i512);
            r512.dispose();
            i512.dispose();
            return c; // [512,2049,2] complex
          });
          const stftB = tf.concat(
            comps.map((c) => c.expandDims(0)),
            0
          ); // [B,512,2049,2]
          comps.forEach((c) => c.dispose());
          tensorsToDispose.push(stftB);
          for (const info of inputsInfo2) {
            if (matchesComplexInput(info)) minimalInputs[info.name] = stftB;
          }
        }
        if (haveMag) {
          const mags = feats.map((f) => {
            const magT = tf.tensor(f.magArr, [f.time, 1024, 2], "float32");
            const magSlice = magT.slice([0, 0, 0], [f.tSpan, 1024, 2]);
            const needPad = 512 - f.tSpan;
            const pad =
              needPad > 0 ? this._getZeroPad(needPad, 1024, 2, "mag") : null;
            const patch = pad ? tf.concat([magSlice, pad], 0) : magSlice;
            magT.dispose();
            // cached pad persists
            magSlice.dispose();
            return patch.expandDims(0); // [1,512,1024,2]
          });
          const magB = tf.concat(mags, 0); // [B,512,1024,2]
          mags.forEach((t) => t.dispose());
          tensorsToDispose.push(magB);
          for (const info of inputsInfo2) {
            if (matchesMag4dInput(info)) minimalInputs[info.name] = magB;
          }
        }
      }
      // Fill other placeholders with zeros/noise
      for (const info of inputsInfo2) {
        const name = info?.name;
        if (!name || minimalInputs[name]) continue;
        const dtype = info.dtype || "float32";
        const shape = info.shape || [];
        const shaped = shape.map((d, idx) =>
          d == null || d < 0 ? (idx === 0 ? B : 1) : d
        );
        let t;
        if (dtype === "string") {
          t = tf.fill(shaped, "");
        } else if (dtype === "complex64") {
          const r = tf.zeros(shaped, "float32");
          const i = tf.zeros(shaped, "float32");
          t = tf.complex(r, i);
          r.dispose();
          i.dispose();
        } else {
          t = this._fillNonAudioWithNoise
            ? tf.randomUniform(shaped, -1e-3, 1e-3, "float32")
            : tf.zeros(
                shaped,
                dtype === "float32" || dtype === "int32" || dtype === "bool"
                  ? dtype
                  : "float32"
              );
        }
        minimalInputs[name] = t;
        tensorsToDispose.push(t);
      }
      // Execute once
      const fetches = this._stemOutputNames(this.numStems);
      let raw = await (this.model.executeAsync
        ? this.model.executeAsync(minimalInputs, fetches)
        : this.model.execute
        ? this.model.execute(minimalInputs, fetches)
        : this.model.predict(minimalInputs));
      const outs = Array.isArray(raw)
        ? raw
        : raw && raw.dtype
        ? [raw]
        : raw && typeof raw === "object"
        ? Object.values(raw)
        : [];
      const numericOuts = outs.filter(
        (t) => t && typeof t.dtype === "string" && t.dtype !== "string"
      );
      // Parse per-batch per-stem
      const S = this.numStems;
      const stemsB = [];
      if (numericOuts.length === S) {
        // Each output is [B, T, 2] or [B, 2, T] etc. Reduce to mono per batch.
        const perOutData = await Promise.all(numericOuts.map((t) => t.data()));
        for (let b = 0; b < B; b++) {
          const perStem = new Array(S).fill(null);
          for (let s = 0; s < S; s++) {
            const t = numericOuts[s];
            const d = perOutData[s];
            const shape = t.shape;
            // Handle [B,T,2] or [B,2,T]
            if (shape.length === 3 && shape[0] === B) {
              if (shape[2] === 2) {
                const Tn = shape[1];
                const base = b * Tn * 2;
                const mono = new Float32Array(Tn);
                for (let tt = 0; tt < Tn; tt++) {
                  const L = d[base + tt * 2 + 0];
                  const R = d[base + tt * 2 + 1];
                  mono[tt] = (L + R) * 0.5;
                }
                perStem[s] = mono;
              } else if (shape[1] === 2) {
                const Tn = shape[2];
                // data per batch contiguous by default in TF.js
                const mono = new Float32Array(Tn);
                const row = b * 2 * Tn;
                for (let tt = 0; tt < Tn; tt++) {
                  const L = d[row + tt];
                  const R = d[row + Tn + tt];
                  mono[tt] = (L + R) * 0.5;
                }
                perStem[s] = mono;
              } else {
                perStem[s] = new Float32Array(firstPrep.T);
              }
            } else {
              perStem[s] = new Float32Array(firstPrep.T);
            }
          }
          stemsB.push(perStem);
        }
      } else {
        // Fallback: per-batch single best output split evenly is non-trivial; fallback to single runs
        for (let b = 0; b < B; b++) {
          const single = await this.runChunk(channelsB[b], sampleRate, opts);
          stemsB.push(single.stems);
        }
      }
      outs.forEach((t) => {
        try {
          t && t.dispose && t.dispose();
        } catch (_) {}
      });
      const seen = new Set();
      tensorsToDispose.forEach((t) => {
        if (!t || !t.dispose) return;
        if (seen.has(t)) return;
        seen.add(t);
        try {
          t.dispose();
        } catch (_) {}
      });
      return { stemsB, backend: this._lastExecBackend };
    } catch (e) {
      // Safe fallback: run singles
      const stemsB = [];
      for (let b = 0; b < B; b++) {
        const single = await this.runChunk(channelsB[b], sampleRate, opts);
        stemsB.push(single.stems);
      }
      return { stemsB, backend: this._lastExecBackend };
    }
  }

  async _runChunkInternal(channelsOrMono, sampleRate, opts = {}) {
    const tf = await ensureTF();
    const tAll0 =
      typeof performance !== "undefined" && performance.now
        ? performance.now()
        : Date.now();
    const dbg = (tag, msg) => {
      try {
        const err = new Error();
        const st = (err.stack || "").split("\n");
        const site = st[2] ? st[2].trim() : "";
        const full = site ? `[${tag} ${site}] ${msg}` : `[${tag}] ${msg}`;
        self &&
          self.postMessage &&
          self.postMessage({ type: "debug", payload: { message: full } });
      } catch (_) {
        try {
          self &&
            self.postMessage &&
            self.postMessage({
              type: "debug",
              payload: { message: `[${tag}] ${msg}` },
            });
        } catch (_) {}
      }
    };

    const channels = Array.isArray(channelsOrMono)
      ? channelsOrMono
      : [channelsOrMono];
    if (!this.model) throw new Error("Model not loaded");

    const inputsInfo = this.model.inputs || [];
    const getRank = (i) => (i && i.shape ? i.shape.length : 0);
    const f32Inputs = inputsInfo.filter((i) => i && i.dtype === "float32");
    let audioInputInfo = null;
    const rank2or3 = f32Inputs.filter((i) => {
      const r = getRank(i);
      return r === 2 || r === 3;
    });
    if (rank2or3.length) audioInputInfo = rank2or3[0];
    else if (f32Inputs.length)
      audioInputInfo = f32Inputs.sort((a, b) => getRank(a) - getRank(b))[0];
    else audioInputInfo = inputsInfo[0] || null;

    const inShape =
      audioInputInfo && audioInputInfo.shape
        ? audioInputInfo.shape.slice()
        : null;
    const rank = inShape ? inShape.length : 3;
    const dims = (inShape || []).map((d) => (d == null || d < 0 ? null : d));
    let Cexp = channels.length;
    let T = channels[0].length;
    let layout = "BTC";
    if (rank === 2) {
      // Decide if shape is [T, C] (TC) or [C, T] (CT)
      const d0 = dims[0];
      const d1 = dims[1];
      if (d1 === channels.length || d1 === 1 || d1 == null) {
        // [T, C]
        layout = "TC";
        T = channels[0].length; // time stays variable (None)
        Cexp = d1 || channels.length;
      } else if (d0 === channels.length || d0 === 1) {
        // [C, T]
        layout = "CT";
        T = channels[0].length;
        Cexp = d0;
      } else {
        // Default to [T, C]
        layout = "TC";
        T = channels[0].length;
        Cexp = channels.length;
      }
    } else if (rank >= 3) {
      const d1 = dims[1];
      const d2 = dims[2];
      if (d2 === channels.length || d2 === 1 || d2 == null) {
        layout = "BTC";
        T = d1 || T;
        Cexp = d2 || channels.length;
      } else if (d1 === channels.length || d1 === 1) {
        layout = "BCT";
        T = dims[2] || T;
        Cexp = d1;
      } else {
        layout = "BTC";
        T = d1 || T;
        Cexp = channels.length;
      }
    }
    if (Cexp > channels.length) {
      while (channels.length < Cexp) channels.push(channels[0]);
    }

    const prepped = channels.slice(0, Cexp).map((ch) => {
      if (ch.length === T) return ch;
      const out = new Float32Array(T);
      out.set(ch.subarray(0, Math.min(ch.length, T)));
      return out;
    });

    const byLayoutTensor = () => {
      if (layout === "TC") {
        const inter = new Float32Array(T * Cexp);
        let idx = 0;
        for (let t = 0; t < T; t++)
          for (let c = 0; c < Cexp; c++) inter[idx++] = prepped[c][t];
        return tf.tensor(inter, [T, Cexp], "float32");
      }
      if (layout === "CT") {
        const byCh = new Float32Array(T * Cexp);
        let idx = 0;
        for (let c = 0; c < Cexp; c++)
          for (let t = 0; t < T; t++) byCh[idx++] = prepped[c][t];
        return tf.tensor(byCh, [Cexp, T], "float32");
      }
      if (layout === "BCT") {
        const byCh = new Float32Array(T * Cexp);
        let idx = 0;
        for (let c = 0; c < Cexp; c++)
          for (let t = 0; t < T; t++) byCh[idx++] = prepped[c][t];
        return tf.tensor(byCh, [1, Cexp, T], "float32");
      }
      const inter = new Float32Array(T * Cexp);
      let idx = 0;
      for (let t = 0; t < T; t++)
        for (let c = 0; c < Cexp; c++) inter[idx++] = prepped[c][t];
      return tf.tensor(inter, [1, T, Cexp], "float32");
    };

    const makeZeros = (shape, dtype) => {
      const sd = (shape || []).map((d, idx) =>
        d == null || d < 0 ? (idx === 0 ? 1 : 1) : d
      );
      const dt = dtype || "float32";
      if (dt === "string") {
        if (sd.length === 0) return tf.scalar("");
        return tf.fill(sd, "");
      }
      if (dt === "complex64") {
        const real =
          sd.length === 0 ? tf.scalar(0, "float32") : tf.zeros(sd, "float32");
        const imag =
          sd.length === 0 ? tf.scalar(0, "float32") : tf.zeros(sd, "float32");
        return tf.complex(real, imag);
      }
      if (sd.length === 0) return tf.scalar(0, dt);
      if (dt === "float32" || dt === "int32" || dt === "bool")
        return tf.zeros(sd, dt);
      return tf.zeros(sd, "float32");
    };
    const makeNoise = (shape, dtype, scale = 1e-3) => {
      const sd = (shape || []).map((d, idx) =>
        d == null || d < 0 ? (idx === 0 ? 1 : 1) : d
      );
      const dt = dtype || "float32";
      if (dt === "string") return makeZeros(sd, dt);
      if (dt === "complex64") {
        const real = tf.randomUniform(sd, -scale, scale, "float32");
        const imag = tf.randomUniform(sd, -scale, scale, "float32");
        return tf.complex(real, imag);
      }
      if (dt === "float32" || dt === "int32") {
        const arr = tf.randomUniform(sd, -scale, scale, "float32");
        if (dt === "int32") return arr.toInt();
        return arr;
      }
      return tf.randomUniform(sd, -scale, scale, "float32");
    };
    const makeAscending = (n) =>
      tf.tensor(
        new Int32Array(Array.from({ length: n }, (_, i) => i)),
        [n],
        "int32"
      );

    const tensorsToDispose = [];
    const tX0 =
      typeof performance !== "undefined" && performance.now
        ? performance.now()
        : Date.now();
    const x = byLayoutTensor();
    const tX1 =
      typeof performance !== "undefined" && performance.now
        ? performance.now()
        : Date.now();
    dbg(
      "inference.js:timing",
      `audio tensor build: ${(tX1 - tX0).toFixed(2)} ms`
    );
    tensorsToDispose.push(x);
    if (!this._loggedFirstInput) {
      const flat = Array.isArray(channels) && channels[0] ? channels[0] : [];
      let min = Infinity,
        max = -Infinity,
        sumSq = 0,
        nz = 0;
      const n = flat.length;
      const maxSamples = 200000;
      const stride = n > maxSamples ? Math.ceil(n / maxSamples) : 1;
      for (let i = 0; i < n; i += stride) {
        const v = flat[i];
        if (v < min) min = v;
        if (v > max) max = v;
        sumSq += v * v;
        if (v !== 0) nz++;
      }
      const rms = Math.sqrt(sumSq / Math.max(1, Math.ceil(n / stride)));
      dbg(
        "inference.js:input",
        `input tensor ${x.shape} layout=${layout} stats:min=${min.toFixed(
          4
        )} max=${max.toFixed(4)} rms=${rms.toFixed(6)} nz=${nz}`
      );
      this._loggedFirstInput = true;
    }
    // Precompute STFT-based features using fast JS FFT (return raw arrays)
    const computeFeatures = async () => {
      const t0 = performance.now();
      const f = this.precomputeFeatures(prepped);
      const t1 = performance.now();
      dbg(
        "inference.js:features",
        `feature compute took ${(t1 - t0).toFixed(2)} ms`
      );
      return f;
    };
    let features = null;
    const tFeat0 =
      typeof performance !== "undefined" && performance.now
        ? performance.now()
        : Date.now();

    const matchesComplexInput = (info) =>
      info &&
      info.dtype === "complex64" &&
      Array.isArray(info.shape) &&
      info.shape.length === 3 &&
      info.shape[2] === 2 &&
      info.shape[1] === 2049;
    const matchesMag4dInput = (info) =>
      info &&
      info.dtype === "float32" &&
      Array.isArray(info.shape) &&
      info.shape.length === 4 &&
      info.shape[1] === 512 &&
      info.shape[2] === 1024 &&
      info.shape[3] === 2;

    const needComplex = (inputsInfo || []).some(matchesComplexInput);
    const needMag4d = (inputsInfo || []).some(matchesMag4dInput);
    if (opts && opts.features) {
      features = opts.features;
    } else if (needComplex || needMag4d) {
      try {
        features = await computeFeatures();
      } catch (e) {
        dbg(
          "inference.js:features",
          `feature computation failed: ${e && e.message ? e.message : e}`
        );
        features = null;
      }
    }
    const tFeat1 =
      typeof performance !== "undefined" && performance.now
        ? performance.now()
        : Date.now();
    dbg(
      "inference.js:timing",
      `features total: ${(tFeat1 - tFeat0).toFixed(2)} ms`
    );

    // Build minimal input dict: audio + strings (defer feature tensors)
    const audioName = audioInputInfo?.name;
    const tMin0 =
      typeof performance !== "undefined" && performance.now
        ? performance.now()
        : Date.now();
    const minimalInputs = {};
    if (audioName) minimalInputs[audioName] = x;
    const stringInputsCreated = [];
    const complexNames = [];
    const mag4dNames = [];
    for (const info of inputsInfo) {
      if (!info?.name || info === audioInputInfo) continue;
      if (info.dtype === "string") {
        const t = makeZeros(info.shape || [], "string");
        minimalInputs[info.name] = t;
        tensorsToDispose.push(t);
        stringInputsCreated.push(info.name);
      } else if (matchesComplexInput(info)) {
        if (features) complexNames.push(info.name);
        else {
          const t = makeZeros(info.shape, info.dtype);
          minimalInputs[info.name] = t;
          tensorsToDispose.push(t);
        }
      } else if (matchesMag4dInput(info)) {
        if (features) mag4dNames.push(info.name);
        else {
          const t = makeZeros(info.shape, "float32");
          minimalInputs[info.name] = t;
          tensorsToDispose.push(t);
        }
      }
    }
    // Helper: materialize feature tensors on the ACTIVE backend into a dict
    const materializeFeaturesInto = (dict, complexList, magList) => {
      if (!features) return;
      let stftStack = null;
      let mag4d = null;
      // Dispose any previous tensors for these keys to avoid cross-backend reuse
      if (complexList && complexList.length) {
        const prev = new Set();
        for (const name of complexList) {
          const v = dict[name];
          if (v && v.dispose) prev.add(v);
        }
        prev.forEach((t) => {
          try {
            t.dispose && t.dispose();
          } catch (_) {}
        });
      }
      if (magList && magList.length) {
        const prev = new Set();
        for (const name of magList) {
          const v = dict[name];
          if (v && v.dispose) prev.add(v);
        }
        prev.forEach((t) => {
          try {
            t.dispose && t.dispose();
          } catch (_) {}
        });
      }
      if (complexList && complexList.length) {
        const realT = tf.tensor(
          features.stftReal,
          [features.time, 2049, 2],
          "float32"
        );
        const imagT = tf.tensor(
          features.stftImag,
          [features.time, 2049, 2],
          "float32"
        );
        const rs = realT.slice([0, 0, 0], [features.tSpan, 2049, 2]);
        const is = imagT.slice([0, 0, 0], [features.tSpan, 2049, 2]);
        const rpad =
          features.tSpan === 512
            ? null
            : tf.zeros([512 - features.tSpan, 2049, 2], "float32");
        const ipad =
          features.tSpan === 512
            ? null
            : tf.zeros([512 - features.tSpan, 2049, 2], "float32");
        const r512 = rpad ? tf.concat([rs, rpad], 0) : rs;
        const i512 = ipad ? tf.concat([is, ipad], 0) : is;
        stftStack = tf.complex(r512, i512); // [512,2049,2]
        realT.dispose();
        imagT.dispose();
        rs.dispose();
        is.dispose();
        if (rpad) rpad.dispose();
        if (ipad) ipad.dispose();
        tensorsToDispose.push(stftStack);
        for (const name of complexList) dict[name] = stftStack;
      }
      if (magList && magList.length) {
        const magT = tf.tensor(
          features.magArr,
          [features.time, 1024, 2],
          "float32"
        );
        const magSlice = magT.slice([0, 0, 0], [features.tSpan, 1024, 2]);
        const pad =
          features.tSpan === 512
            ? null
            : tf.zeros([512 - features.tSpan, 1024, 2]);
        const magPatch = pad ? tf.concat([magSlice, pad], 0) : magSlice;
        mag4d = magPatch.expandDims(0);
        magT.dispose();
        magSlice.dispose();
        if (pad) pad.dispose();
        tensorsToDispose.push(mag4d);
        for (const name of magList) dict[name] = mag4d;
      }
    };
    const tMin1 =
      typeof performance !== "undefined" && performance.now
        ? performance.now()
        : Date.now();
    dbg(
      "inference.js:timing",
      `minimal inputs build: ${(tMin1 - tMin0).toFixed(2)} ms`
    );
    const allComplexNames = [];
    const allMag4dNames = [];
    const buildAllInputs = async () => {
      const dict = {};
      for (const info of inputsInfo) {
        const name = info && info.name;
        if (!name) continue;
        if (info === audioInputInfo) {
          dict[name] = x;
          continue;
        }
        const dtype = info.dtype || "float32";
        const shape = info.shape || [];
        if (dtype === "string") {
          const t = makeZeros(shape, "string");
          dict[name] = t;
          tensorsToDispose.push(t);
          continue;
        }
        if (matchesComplexInput(info)) {
          if (features && features.stftReal && features.stftImag) {
            allComplexNames.push(name);
          } else {
            const t = makeZeros(shape, dtype);
            dict[name] = t;
            tensorsToDispose.push(t);
          }
          continue;
        }
        if (matchesMag4dInput(info)) {
          if (features && features.magArr) {
            allMag4dNames.push(name);
          } else {
            const t = makeZeros(shape, "float32");
            dict[name] = t;
            tensorsToDispose.push(t);
          }
          continue;
        }
        if (shape.length === 4) {
          const sd = shape.map((d, idx) =>
            typeof d === "number" && d > 0 ? d : idx === 0 ? 1 : 1
          );
          const t = this._fillNonAudioWithNoise
            ? makeNoise(sd, dtype)
            : makeZeros(sd, dtype);
          dict[name] = t;
          tensorsToDispose.push(t);
          continue;
        }
        if (
          dtype === "int32" &&
          shape &&
          shape.length === 1 &&
          typeof shape[0] === "number" &&
          shape[0] > 0
        ) {
          const t = makeAscending(shape[0]);
          dict[name] = t;
          tensorsToDispose.push(t);
        } else {
          const t = this._fillNonAudioWithNoise
            ? makeNoise(shape, dtype)
            : makeZeros(shape, dtype);
          dict[name] = t;
          tensorsToDispose.push(t);
        }
      }
      return dict;
    };
    // Helper to run model with timeout to avoid indefinite GPU hangs
    const doExec = (dict, fetches) =>
      this.model.executeAsync
        ? this.model.executeAsync(dict, fetches)
        : this.model.execute
        ? this.model.execute(dict, fetches)
        : this.model.predict(dict);
    const withTimeout = async (factory, ms, label) => {
      let to;
      const timeout = new Promise((_, rej) => {
        to = setTimeout(() => rej(new Error(`exec timeout (${label})`)), ms);
      });
      try {
        return await Promise.race([factory(), timeout]);
      } finally {
        clearTimeout(to);
      }
    };
    const EXEC_TIMEOUT_MS = 6000;
    // Log which input we chose for audio and what we fed others (once)
    if (!this._loggedInputsOnce) {
      const audioName = audioInputInfo?.name || "<unknown>";
      const details = (inputsInfo || [])
        .map((i) => `${i?.name}:{${i?.dtype}}${JSON.stringify(i?.shape || [])}`)
        .join(", ");
      dbg(
        "inference.js:inputs",
        `audioInput=${audioName} layout=${layout}; allInputs=[${details}] fillerNoise=${this._fillNonAudioWithNoise}`
      );
      this._loggedInputsOnce = true;
    }

    let raw;
    let usedPath = "minimal";
    let usedInputsRef = minimalInputs;
    try {
      const hasComplexInputs = (this.model.inputs || []).some(
        (i) => i && i.dtype === "complex64"
      );
      const fetches = this._stemOutputNames(this.numStems);
      const tExec0 =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      let triedWebGL = false;
      let execBackend = tf.getBackend();
      const prevBackend = execBackend;
      // Try switch to WebGL just for model execution (unless forceCpu or lockBackend)
      try {
        const allowWebglOverride =
          prevBackend === "wasm" && hasComplexInputs === true;
        if (opts?.forceCpu || (opts?.lockBackend && !allowWebglOverride)) {
          triedWebGL = false;
          execBackend = tf.getBackend();
          throw new Error("forceCpu enabled");
        }
        if (prevBackend !== "webgl") {
          try {
            await tf.setBackend("webgl");
          } catch (eSet) {
            dbg(
              "inference.js:exec",
              `setBackend(webgl) threw: ${
                eSet && eSet.message ? eSet.message : eSet
              }`
            );
            throw eSet;
          }
          await tf.ready();
          if (tf.getBackend() === "webgl") {
            triedWebGL = true;
            execBackend = "webgl";
            try {
              // Log active backend
              dbg("inference.js:exec", `webgl active`);
            } catch (_) {}
          } else {
            dbg(
              "inference.js:exec",
              `webgl not active after set; backend=${tf.getBackend()}`
            );
            await tf.setBackend(prevBackend);
            await tf.ready();
            execBackend = prevBackend;
          }
        }
      } catch (eSwitch) {
        dbg(
          "inference.js:exec",
          `webgl switch failed; staying on ${prevBackend}. Reason: ${
            eSwitch && eSwitch.message ? eSwitch.message : eSwitch
          }`
        );
        if (opts?.forceCpu) {
          try {
            await tf.setBackend("cpu");
            await tf.ready();
            execBackend = "cpu";
          } catch (_) {
            execBackend = tf.getBackend();
          }
        } else {
          try {
            await tf.setBackend(prevBackend);
            await tf.ready();
          } catch (_) {}
          execBackend = prevBackend;
        }
      }
      // If we have deferred feature tensors to fill, materialize them now on the active backend
      if (features && (complexNames.length || mag4dNames.length)) {
        materializeFeaturesInto(minimalInputs, complexNames, mag4dNames);
      }
      if (fetches && fetches.length) {
        dbg(
          "inference.js:exec",
          `execute minimal+fetches (backend=${tf.getBackend()})`
        );
        try {
          raw = await withTimeout(
            () => doExec(minimalInputs, fetches),
            EXEC_TIMEOUT_MS,
            "webgl:minimal+fetches"
          );
          usedPath = "minimal+fetches";
        } catch (ex) {
          dbg(
            "inference.js:exec",
            `webgl execute failed (minimal+fetches): ${
              ex && ex.message ? ex.message : ex
            }`
          );
          if (triedWebGL) {
            try {
              await tf.setBackend("cpu");
              await tf.ready();
            } catch (_) {}
            // Rebuild feature tensors on CPU backend if needed
            if (features && (complexNames.length || mag4dNames.length)) {
              materializeFeaturesInto(minimalInputs, complexNames, mag4dNames);
            }
            const tRetry0 =
              typeof performance !== "undefined" && performance.now
                ? performance.now()
                : Date.now();
            dbg("inference.js:exec", `retry minimal+fetches on cpu`);
            raw = await doExec(minimalInputs, fetches);
            const tRetry1 =
              typeof performance !== "undefined" && performance.now
                ? performance.now()
                : Date.now();
            dbg(
              "inference.js:timing",
              `model exec retry (cpu): ${(tRetry1 - tRetry0).toFixed(2)} ms`
            );
            execBackend = "cpu";
          } else {
            throw ex;
          }
        }
      } else {
        dbg(
          "inference.js:exec",
          `execute minimal (backend=${tf.getBackend()})`
        );
        // Materialize features if needed for minimal path
        if (features && (complexNames.length || mag4dNames.length)) {
          materializeFeaturesInto(minimalInputs, complexNames, mag4dNames);
        }
        try {
          raw = await withTimeout(
            () => doExec(minimalInputs),
            EXEC_TIMEOUT_MS,
            "webgl:minimal"
          );
        } catch (ex) {
          dbg(
            "inference.js:exec",
            `webgl execute failed (minimal): ${
              ex && ex.message ? ex.message : ex
            }`
          );
          if (triedWebGL) {
            try {
              await tf.setBackend("cpu");
              await tf.ready();
            } catch (_) {}
            if (features && (complexNames.length || mag4dNames.length)) {
              materializeFeaturesInto(minimalInputs, complexNames, mag4dNames);
            }
            const tRetry0 =
              typeof performance !== "undefined" && performance.now
                ? performance.now()
                : Date.now();
            dbg("inference.js:exec", `retry minimal on cpu`);
            raw = await doExec(minimalInputs);
            const tRetry1 =
              typeof performance !== "undefined" && performance.now
                ? performance.now()
                : Date.now();
            dbg(
              "inference.js:timing",
              `model exec retry (cpu): ${(tRetry1 - tRetry0).toFixed(2)} ms`
            );
            execBackend = "cpu";
          } else {
            throw ex;
          }
        }
      }
      // Restore previous backend (e.g., WASM) after exec when we temporarily switched to WebGL
      try {
        if (triedWebGL && prevBackend !== "webgl") {
          await tf.setBackend(prevBackend);
          await tf.ready();
        }
      } catch (_) {}
      const tExec1 =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      this._lastExecBackend = execBackend;
      dbg(
        "inference.js:timing",
        `model exec (${usedPath}, backend=${execBackend}): ${(
          tExec1 - tExec0
        ).toFixed(2)} ms`
      );
    } catch (e) {
      // Fall back to feeding all placeholders
      usedPath = "all";
      const tAllIn0 =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      // Ensure backend compatibility for complex64 placeholders on WASM
      const hasComplexInputsB = (this.model.inputs || []).some(
        (i) => i && i.dtype === "complex64"
      );
      let prevBackendForComplexB = null;
      try {
        if (tf.getBackend() === "wasm" && hasComplexInputsB) {
          prevBackendForComplexB = "wasm";
          await tf.setBackend("cpu");
          await tf.ready();
        }
      } catch (_) {}
      const inputsDict = await buildAllInputs();
      const tAllIn1 =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      dbg(
        "inference.js:timing",
        `all inputs build: ${(tAllIn1 - tAllIn0).toFixed(2)} ms`
      );
      usedInputsRef = inputsDict;
      const fetches = this._stemOutputNames(this.numStems);
      const tExec0 =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      let triedWebGL = false;
      let execBackend = tf.getBackend();
      const prevBackend = execBackend;
      try {
        const allowWebglOverrideB =
          prevBackend === "wasm" &&
          (this.model.inputs || []).some((i) => i && i.dtype === "complex64");
        if (opts?.forceCpu || (opts?.lockBackend && !allowWebglOverrideB)) {
          triedWebGL = false;
          execBackend = tf.getBackend();
          throw new Error("forceCpu enabled");
        }
        if (prevBackend !== "webgl") {
          try {
            await tf.setBackend("webgl");
          } catch (eSet) {
            dbg(
              "inference.js:exec",
              `setBackend(webgl) threw: ${
                eSet && eSet.message ? eSet.message : eSet
              }`
            );
            throw eSet;
          }
          await tf.ready();
          if (tf.getBackend() === "webgl") {
            triedWebGL = true;
            execBackend = "webgl";
          } else {
            dbg(
              "inference.js:exec",
              `webgl not active after set; backend=${tf.getBackend()}`
            );
            await tf.setBackend(prevBackend);
            await tf.ready();
            execBackend = prevBackend;
          }
        }
      } catch (eSwitch) {
        dbg(
          "inference.js:exec",
          `webgl switch failed; staying on ${prevBackend}. Reason: ${
            eSwitch && eSwitch.message ? eSwitch.message : eSwitch
          }`
        );
        if (opts?.forceCpu) {
          try {
            await tf.setBackend("cpu");
            await tf.ready();
            execBackend = "cpu";
          } catch (_) {
            execBackend = tf.getBackend();
          }
        } else {
          try {
            await tf.setBackend(prevBackend);
            await tf.ready();
          } catch (_) {}
          execBackend = prevBackend;
        }
      }
      // Materialize all-input features if needed on active backend
      if (features && (allComplexNames.length || allMag4dNames.length)) {
        materializeFeaturesInto(inputsDict, allComplexNames, allMag4dNames);
      }
      if (fetches && fetches.length) {
        dbg(
          "inference.js:exec",
          `execute all+fetches (backend=${tf.getBackend()})`
        );
        try {
          raw = await withTimeout(
            () => doExec(inputsDict, fetches),
            EXEC_TIMEOUT_MS,
            "webgl:all+fetches"
          );
          usedPath = "all+fetches";
        } catch (ex) {
          dbg(
            "inference.js:exec",
            `webgl execute failed (all+fetches): ${
              ex && ex.message ? ex.message : ex
            }`
          );
          if (triedWebGL) {
            try {
              await tf.setBackend("cpu");
              await tf.ready();
            } catch (_) {}
            // Rebuild feature tensors on CPU if needed
            if (features && (allComplexNames.length || allMag4dNames.length)) {
              materializeFeaturesInto(
                inputsDict,
                allComplexNames,
                allMag4dNames
              );
            }
            const tRetry0 =
              typeof performance !== "undefined" && performance.now
                ? performance.now()
                : Date.now();
            dbg("inference.js:exec", `retry all+fetches on cpu`);
            raw = await doExec(inputsDict, fetches);
            const tRetry1 =
              typeof performance !== "undefined" && performance.now
                ? performance.now()
                : Date.now();
            dbg(
              "inference.js:timing",
              `model exec retry (cpu): ${(tRetry1 - tRetry0).toFixed(2)} ms`
            );
            execBackend = "cpu";
          } else {
            throw ex;
          }
        }
      } else {
        dbg("inference.js:exec", `execute all (backend=${tf.getBackend()})`);
        if (features && (allComplexNames.length || allMag4dNames.length)) {
          materializeFeaturesInto(inputsDict, allComplexNames, allMag4dNames);
        }
        try {
          raw = await withTimeout(
            () => doExec(inputsDict),
            EXEC_TIMEOUT_MS,
            "webgl:all"
          );
        } catch (ex) {
          dbg(
            "inference.js:exec",
            `webgl execute failed (all): ${ex && ex.message ? ex.message : ex}`
          );
          if (triedWebGL) {
            try {
              await tf.setBackend("cpu");
              await tf.ready();
            } catch (_) {}
            if (features && (allComplexNames.length || allMag4dNames.length)) {
              materializeFeaturesInto(
                inputsDict,
                allComplexNames,
                allMag4dNames
              );
            }
            const tRetry0 =
              typeof performance !== "undefined" && performance.now
                ? performance.now()
                : Date.now();
            dbg("inference.js:exec", `retry all on cpu`);
            raw = await doExec(inputsDict);
            const tRetry1 =
              typeof performance !== "undefined" && performance.now
                ? performance.now()
                : Date.now();
            dbg(
              "inference.js:timing",
              `model exec retry (cpu): ${(tRetry1 - tRetry0).toFixed(2)} ms`
            );
            execBackend = "cpu";
          } else {
            throw ex;
          }
        }
      }
      // Restore to previous backend after exec (e.g., WASM)
      try {
        if (triedWebGL && prevBackend !== "webgl") {
          await tf.setBackend(prevBackend);
          await tf.ready();
        }
      } catch (_) {}
      // Restore WASM if we temporarily switched due to complex inputs
      try {
        if (prevBackendForComplexB === "wasm") {
          await tf.setBackend("wasm");
          await tf.ready();
        }
      } catch (_) {}
      try {
        if (triedWebGL && !opts?.stickyBackend && !opts?.lockBackend) {
          await tf.setBackend("cpu");
          await tf.ready();
        }
      } catch (_) {}
      const tExec1 =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      this._lastExecBackend = execBackend;
      dbg(
        "inference.js:timing",
        `model exec (${usedPath}, backend=${execBackend}): ${(
          tExec1 - tExec0
        ).toFixed(2)} ms`
      );
    }
    if (!this._loggedPath) {
      dbg(
        "inference.js:exec",
        `execute path: ${usedPath} (strings: ${stringInputsCreated.join(",")})`
      );
      this._loggedPath = true;
    }
    if (!this._loggedFedOnce) {
      try {
        const fed = Object.entries(usedInputsRef || {}).map(([k, v]) => {
          const dt = v && v.dtype ? v.dtype : typeof v;
          const sh = v && v.shape ? JSON.stringify(v.shape) : "[]";
          return `${k}:{${dt}}${sh}`;
        });
        dbg("inference.js:fed", `inputs=[${fed.join(", ")}]`);
      } catch (_) {}
      this._loggedFedOnce = true;
    }
    const tParse0 =
      typeof performance !== "undefined" && performance.now
        ? performance.now()
        : Date.now();
    const outs = Array.isArray(raw)
      ? raw
      : raw && raw.dtype
      ? [raw]
      : raw && typeof raw === "object"
      ? Object.values(raw)
      : [];
    const numericOuts = outs.filter(
      (t) => t && typeof t.dtype === "string" && t.dtype !== "string"
    );

    const parseSingle = (t) => {
      const shape = t.shape;
      const data = t.dataSync();
      const S = this.numStems;
      if (shape.length === 3) {
        const d1 = shape[1];
        const d2 = shape[2];
        if (d2 === S) {
          const Tn = d1;
          const stems = new Array(S).fill(null).map(() => new Float32Array(Tn));
          let idx = 0;
          for (let tt = 0; tt < Tn; tt++)
            for (let s = 0; s < S; s++) stems[s][tt] = data[idx++];
          return stems;
        } else if (d1 === S) {
          const Tn = d2;
          const stems = new Array(S).fill(null).map(() => new Float32Array(Tn));
          let idx = 0;
          for (let s = 0; s < S; s++)
            for (let tt = 0; tt < Tn; tt++) stems[s][tt] = data[idx++];
          return stems;
        }
      } else if (shape.length === 2) {
        const d1 = shape[0];
        const d2 = shape[1];
        if (d1 === S) {
          const Tn = d2;
          const stems = new Array(S).fill(null).map(() => new Float32Array(Tn));
          let idx = 0;
          for (let s = 0; s < S; s++)
            for (let tt = 0; tt < Tn; tt++) stems[s][tt] = data[idx++];
          return stems;
        } else if (d2 === S) {
          const Tn = d1;
          const stems = new Array(S).fill(null).map(() => new Float32Array(Tn));
          let idx = 0;
          for (let tt = 0; tt < Tn; tt++)
            for (let s = 0; s < S; s++) stems[s][tt] = data[idx++];
          return stems;
        }
      }
      return new Array(S).fill(null).map(() => new Float32Array(T));
    };

    const stemRms = (a) => {
      let s = 0;
      const n = a.length;
      if (!n) return 0;
      const maxSamples = 200000;
      const stride = n > maxSamples ? Math.ceil(n / maxSamples) : 1;
      let count = 0;
      for (let i = 0; i < n; i += stride) {
        const v = a[i];
        s += v * v;
        count++;
      }
      return Math.sqrt(s / Math.max(1, count));
    };
    const toMonoFrom2D = (t) => {
      const shape = t.shape;
      const data = t.dataSync();
      if (shape.length === 2) {
        const d0 = shape[0],
          d1 = shape[1];
        if (d1 === 2) {
          const Tn = d0;
          const out = new Float32Array(Tn);
          let idx = 0;
          for (let tt = 0; tt < Tn; tt++) {
            const L = data[idx++];
            const R = data[idx++];
            out[tt] = (L + R) * 0.5;
          }
          return out;
        } else if (d0 === 2) {
          const Tn = d1;
          const out = new Float32Array(Tn);
          let idx = 0;
          // two rows of length Tn: [2, T]
          for (let tt = 0; tt < Tn; tt++) out[tt] = data[tt];
          for (let tt = 0; tt < Tn; tt++)
            out[tt] = (out[tt] + data[Tn + tt]) * 0.5;
          return out;
        }
      } else if (shape.length === 3 && shape[0] === 1) {
        // [1, T, 2] or [1, 2, T]
        if (shape[2] === 2) {
          const Tn = shape[1];
          const out = new Float32Array(Tn);
          let idx = 0;
          for (let tt = 0; tt < Tn; tt++) {
            const L = data[idx++];
            const R = data[idx++];
            out[tt] = (L + R) * 0.5;
          }
          return out;
        } else if (shape[1] === 2) {
          const Tn = shape[2];
          // data layout may be [1,2,T]; average second dim
          const out = new Float32Array(Tn);
          for (let tt = 0; tt < Tn; tt++) {
            const L = data[tt];
            const R = data[Tn + tt];
            out[tt] = (L + R) * 0.5;
          }
          return out;
        }
      }
      // fallback: flatten to mono
      return new Float32Array(T);
    };

    let stems;
    if (
      (usedPath === "minimal+fetches" || usedPath === "all+fetches") &&
      numericOuts.length === this.numStems
    ) {
      // We fetched one tensor per stem; assemble stems in order
      const logs = [];
      stems = new Array(this.numStems).fill(null).map((_, i) => {
        const t = numericOuts[i];
        const mono = toMonoFrom2D(t);
        logs.push(
          `out[${i}] ${t.dtype} ${JSON.stringify(t.shape)} rms=${stemRms(
            mono
          ).toFixed(6)}`
        );
        return mono;
      });
      if (!this._loggedFirstOutputs) {
        dbg(
          "inference.js:outputs",
          `outputs (mapped per-stem): ${logs.join(" | ")}`
        );
        this._loggedFirstOutputs = true;
      }
    } else if (numericOuts.length > 0) {
      // Fallback: choose the most informative output and parse into stems
      let best = { rms: -1, idx: -1, stems: null };
      const logs = [];
      for (let i = 0; i < numericOuts.length; i++) {
        const t = numericOuts[i];
        const ss = parseSingle(t);
        const total = ss.reduce((acc, s) => acc + stemRms(s), 0);
        logs.push(
          `out[${i}] ${t.dtype} ${JSON.stringify(
            t.shape
          )} totalRMS=${total.toFixed(6)}`
        );
        if (total > best.rms) best = { rms: total, idx: i, stems: ss };
      }
      if (!this._loggedFirstOutputs) {
        dbg("inference.js:outputs", `outputs: ${logs.join(" | ")}`);
        this._loggedFirstOutputs = true;
      }
      stems =
        best.stems ||
        new Array(this.numStems).fill(null).map(() => new Float32Array(T));
    } else {
      dbg(
        "inference.js:outputs",
        `no numeric outputs (outs=${outs.length}, numeric=${numericOuts.length})`
      );
      stems = new Array(this.numStems)
        .fill(null)
        .map(() => new Float32Array(T));
    }
    const tParse1 =
      typeof performance !== "undefined" && performance.now
        ? performance.now()
        : Date.now();
    dbg(
      "inference.js:timing",
      `outputs parse: ${(tParse1 - tParse0).toFixed(2)} ms`
    );

    const tDisp0 =
      typeof performance !== "undefined" && performance.now
        ? performance.now()
        : Date.now();
    outs.forEach((t) => {
      try {
        t && t.dispose && t.dispose();
      } catch (_) {}
    });
    const seen = new Set();
    tensorsToDispose.forEach((t) => {
      if (!t || !t.dispose) return;
      if (seen.has(t)) return;
      seen.add(t);
      try {
        t.dispose();
      } catch (_) {}
    });
    const tDisp1 =
      typeof performance !== "undefined" && performance.now
        ? performance.now()
        : Date.now();
    dbg("inference.js:timing", `dispose: ${(tDisp1 - tDisp0).toFixed(2)} ms`);
    const tAll1 =
      typeof performance !== "undefined" && performance.now
        ? performance.now()
        : Date.now();
    dbg(
      "inference.js:timing",
      `runChunk total: ${(tAll1 - tAll0).toFixed(2)} ms`
    );
    return { stems, backend: this._lastExecBackend };
  }
}
