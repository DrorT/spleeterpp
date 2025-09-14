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
    try {
      if (tf.getBackend && tf.getBackend() !== "cpu") {
        await tf.setBackend("cpu");
        await tf.ready();
      }
    } catch (_) {}
    // Prefer CPU for stability (feature compute + model exec in same backend)
    try {
      await configureBackend("cpu");
    } catch (_) {}
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

  async runChunk(channelsOrMono, sampleRate) {
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
      const backend = tf.getBackend();
      if (backend === "webgl") {
        dbg("inference.js:features", "backend=webgl; skipping feature compute");
        return null;
      }
      dbg("inference.js:features", `start (backend=${backend})`);
      const t0 = performance.now();
      // Compute STFT with custom FFT for up to 2 channels
      const { channels: st, frames, bins } = stftStereo(prepped, 1024);
      let time = frames;
      if (!time || time <= 0) time = 1; // guard against tiny chunks
      // Build real/imag arrays shaped [frames, bins(2049), 2]
      const ch0 = st[0];
      const ch1 = st.length > 1 ? st[1] : st[0];
      const outBins = 2049; // bins
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
      // Magnitude patch arrays [time,1024,2] -> later [1,512,1024,2]
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
          // log1p magnitude scaling improves numerical range for models
          magArr[dstBase + k * 2 + 0] = Math.log1p(Math.hypot(r0, i0v));
          magArr[dstBase + k * 2 + 1] = Math.log1p(Math.hypot(r1, i1v));
        }
      }
      const tSpan = Math.min(512, time);
      // JS stats only (no tensors yet)
      let minv = Infinity,
        maxv = -Infinity,
        sum = 0,
        cnt = 0;
      for (let i = 0; i < magArr.length; i++) {
        const v = magArr[i];
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
        sum += v;
        cnt++;
      }
      const t1 = performance.now();
      dbg(
        "inference.js:features",
        `feature compute took ${(t1 - t0).toFixed(2)} ms`
      );
      dbg(
        "inference.js:features",
        `done frames=${time} bins=${outBins} mag stats: min=${minv.toFixed(
          4
        )} max=${maxv.toFixed(4)} mean=${(sum / Math.max(1, cnt)).toFixed(6)}`
      );
      return {
        time,
        outBins,
        tSpan,
        stftReal: realArr,
        stftImag: imagArr,
        magArr,
      };
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
    if (needComplex || needMag4d) {
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
        stftStack = tf.complex(realT, imagT);
        realT.dispose();
        imagT.dispose();
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
      const fetches = this._stemOutputNames(this.numStems);
      const tExec0 =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
      let triedWebGL = false;
      let execBackend = tf.getBackend();
      const prevBackend = execBackend;
      // Try switch to WebGL just for model execution
      try {
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
        try {
          await tf.setBackend(prevBackend);
          await tf.ready();
        } catch (_) {}
        execBackend = prevBackend;
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
          raw = this.model.executeAsync
            ? await this.model.executeAsync(minimalInputs, fetches)
            : this.model.execute
            ? this.model.execute(minimalInputs, fetches)
            : this.model.predict(minimalInputs);
          usedPath = "minimal+fetches";
        } catch (ex) {
          dbg(
            "inference.js:exec",
            `webgl execute failed (minimal+fetches): ${
              ex && ex.message ? ex.message : ex
            }`
          );
          // If WebGL failed, fall back to CPU
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
            raw = this.model.executeAsync
              ? await this.model.executeAsync(minimalInputs, fetches)
              : this.model.execute
              ? this.model.execute(minimalInputs, fetches)
              : this.model.predict(minimalInputs);
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
          raw = this.model.executeAsync
            ? await this.model.executeAsync(minimalInputs)
            : this.model.execute
            ? this.model.execute(minimalInputs)
            : this.model.predict(minimalInputs);
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
            raw = this.model.executeAsync
              ? await this.model.executeAsync(minimalInputs)
              : this.model.execute
              ? this.model.execute(minimalInputs)
              : this.model.predict(minimalInputs);
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
      // Restore to CPU after exec if we switched
      try {
        if (triedWebGL) {
          await tf.setBackend("cpu");
          await tf.ready();
        }
      } catch (_) {}
      const tExec1 =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
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
        try {
          await tf.setBackend(prevBackend);
          await tf.ready();
        } catch (_) {}
        execBackend = prevBackend;
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
          raw = this.model.executeAsync
            ? await this.model.executeAsync(inputsDict, fetches)
            : this.model.execute
            ? this.model.execute(inputsDict, fetches)
            : this.model.predict(inputsDict);
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
            raw = this.model.executeAsync
              ? await this.model.executeAsync(inputsDict, fetches)
              : this.model.execute
              ? this.model.execute(inputsDict, fetches)
              : this.model.predict(inputsDict);
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
          raw = this.model.executeAsync
            ? await this.model.executeAsync(inputsDict)
            : this.model.execute
            ? this.model.execute(inputsDict)
            : this.model.predict(inputsDict);
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
            raw = this.model.executeAsync
              ? await this.model.executeAsync(inputsDict)
              : this.model.execute
              ? this.model.execute(inputsDict)
              : this.model.predict(inputsDict);
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
      try {
        if (triedWebGL) {
          await tf.setBackend("cpu");
          await tf.ready();
        }
      } catch (_) {}
      const tExec1 =
        typeof performance !== "undefined" && performance.now
          ? performance.now()
          : Date.now();
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
    return { stems };
  }
}
