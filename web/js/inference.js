// Inference adapter for TF.js models (model loading + inference)
import { ensureTF } from "./tf-backend.js";
import { loadGraphModelWithCache } from "./model-loader.js";

export class InferenceEngine {
  constructor(opts = {}) {
    this.numStems = opts.numStems || 2;
    this.model = null;
    this.modelUrl = null;
    this.io = { inputs: [], outputs: [] };
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
      T = dims[1] || T;
      Cexp = 1;
      layout = "BT";
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
      if (layout === "BT") return tf.tensor(prepped[0], [1, T], "float32");
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
    const makeAscending = (n) =>
      tf.tensor(
        new Int32Array(Array.from({ length: n }, (_, i) => i)),
        [n],
        "int32"
      );

    const tensorsToDispose = [];
    const x = byLayoutTensor();
    tensorsToDispose.push(x);
    const inputsDict = {};
    for (const info of inputsInfo) {
      const name = info && info.name;
      if (!name) continue;
      if (info === audioInputInfo) {
        inputsDict[name] = x;
      } else {
        const dtype = info.dtype || "float32";
        const shape = info.shape || [];
        if (dtype === "string") {
          const t = makeZeros(shape, "string");
          inputsDict[name] = t;
          tensorsToDispose.push(t);
          continue;
        }
        if (shape.length === 4) {
          const sd = shape.map((d, idx) =>
            typeof d === "number" && d > 0 ? d : idx === 0 ? 1 : 1
          );
          const t = makeZeros(sd, dtype);
          inputsDict[name] = t;
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
          inputsDict[name] = t;
          tensorsToDispose.push(t);
        } else {
          const t = makeZeros(shape, dtype);
          inputsDict[name] = t;
          tensorsToDispose.push(t);
        }
      }
    }

    const raw = this.model.executeAsync
      ? await this.model.executeAsync(inputsDict)
      : this.model.execute
      ? this.model.execute(inputsDict)
      : this.model.predict(inputsDict);
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
      return new Array(S).fill(null).map(() => prepped[0].slice());
    };

    let stems;
    if (numericOuts.length === 1) stems = parseSingle(numericOuts[0]);
    else if (numericOuts.length >= this.numStems) {
      stems = [];
      for (let i = 0; i < this.numStems; i++) {
        const t = numericOuts[i];
        const data = t.dataSync();
        stems.push(new Float32Array(data));
      }
    } else {
      stems = new Array(this.numStems).fill(null).map(() => prepped[0].slice());
    }

    outs.forEach((t) => t.dispose && t.dispose());
    tensorsToDispose.forEach((t) => t && t.dispose && t.dispose());
    return { stems };
  }
}
