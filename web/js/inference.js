// Inference adapter for TF.js models (model loading + inference)
import { ensureTF } from "./tf-backend.js";
import { loadGraphModelWithCache } from "./model-loader.js";

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
    const x = byLayoutTensor();
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
      self &&
        self.postMessage &&
        self.postMessage({
          type: "debug",
          payload: {
            message: `input tensor ${
              x.shape
            } layout=${layout} stats:min=${min.toFixed(4)} max=${max.toFixed(
              4
            )} rms=${rms.toFixed(6)} nz=${nz}`,
          },
        });
      this._loggedFirstInput = true;
    }
    // Build minimal input dict: audio + any string placeholders
    const audioName = audioInputInfo?.name;
    const minimalInputs = {};
    if (audioName) minimalInputs[audioName] = x;
    const stringInputsCreated = [];
    for (const info of inputsInfo) {
      if (!info?.name || info === audioInputInfo) continue;
      if (info.dtype === "string") {
        const t = makeZeros(info.shape || [], "string");
        minimalInputs[info.name] = t;
        tensorsToDispose.push(t);
        stringInputsCreated.push(info.name);
      }
    }
    const buildAllInputs = () => {
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
      self &&
        self.postMessage &&
        self.postMessage({
          type: "debug",
          payload: {
            message: `audioInput=${audioName} layout=${layout}; allInputs=[${details}] fillerNoise=${this._fillNonAudioWithNoise}`,
          },
        });
      this._loggedInputsOnce = true;
    }

    let raw;
    let usedPath = "minimal";
    try {
      const fetches = this._stemOutputNames(this.numStems);
      if (fetches && fetches.length) {
        raw = this.model.executeAsync
          ? await this.model.executeAsync(minimalInputs, fetches)
          : this.model.execute
          ? this.model.execute(minimalInputs, fetches)
          : this.model.predict(minimalInputs);
        usedPath = "minimal+fetches";
      } else {
        raw = this.model.executeAsync
          ? await this.model.executeAsync(minimalInputs)
          : this.model.execute
          ? this.model.execute(minimalInputs)
          : this.model.predict(minimalInputs);
      }
    } catch (e) {
      // Fall back to feeding all placeholders
      usedPath = "all";
      const inputsDict = buildAllInputs();
      const fetches = this._stemOutputNames(this.numStems);
      if (fetches && fetches.length) {
        raw = this.model.executeAsync
          ? await this.model.executeAsync(inputsDict, fetches)
          : this.model.execute
          ? this.model.execute(inputsDict, fetches)
          : this.model.predict(inputsDict);
        usedPath = "all+fetches";
      } else {
        raw = this.model.executeAsync
          ? await this.model.executeAsync(inputsDict)
          : this.model.execute
          ? this.model.execute(inputsDict)
          : this.model.predict(inputsDict);
      }
    }
    if (!this._loggedPath && self && self.postMessage) {
      self.postMessage({
        type: "debug",
        payload: {
          message: `execute path: ${usedPath} (strings: ${stringInputsCreated.join(
            ","
          )})`,
        },
      });
      this._loggedPath = true;
    }
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
        const d1 = shape[1],
          d2 = shape[2];
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
        const d1 = shape[0],
          d2 = shape[1];
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
    if (numericOuts.length > 0) {
      let best = { rms: -1, idx: -1, stems: null };
      const logs = [];
      for (let i = 0; i < numericOuts.length; i++) {
        const t = numericOuts[i];
        // If we used fetch names and count equals stems, convert each to mono
        let ss;
        if (
          (usedPath === "minimal+fetches" || usedPath === "all+fetches") &&
          numericOuts.length === this.numStems
        ) {
          ss = new Array(this.numStems)
            .fill(null)
            .map((_, k) => (k === i ? toMonoFrom2D(t) : new Float32Array(T)));
        } else {
          ss = parseSingle(t);
        }
        const total = ss.reduce((acc, s) => acc + stemRms(s), 0);
        logs.push(
          `out[${i}] ${t.dtype} ${JSON.stringify(
            t.shape
          )} totalRMS=${total.toFixed(6)}`
        );
        if (total > best.rms) best = { rms: total, idx: i, stems: ss };
      }
      if (!this._loggedFirstOutputs && self && self.postMessage) {
        self.postMessage({
          type: "debug",
          payload: { message: `outputs: ${logs.join(" | ")}` },
        });
        this._loggedFirstOutputs = true;
      }
      stems =
        best.stems ||
        new Array(this.numStems).fill(null).map(() => new Float32Array(T));
    } else {
      self &&
        self.postMessage &&
        self.postMessage({
          type: "debug",
          payload: {
            message: `no numeric outputs (outs=${outs.length}, numeric=${numericOuts.length})`,
          },
        });
      stems = new Array(this.numStems)
        .fill(null)
        .map(() => new Float32Array(T));
    }

    outs.forEach((t) => t.dispose && t.dispose());
    tensorsToDispose.forEach((t) => t && t.dispose && t.dispose());
    return { stems };
  }
}
