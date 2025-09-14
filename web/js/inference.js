// Inference adapter for TF.js models (model loading implemented; inference TBD)
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

  async runChunk(chunkPCM /* Float32Array */, sampleRate) {
    // TODO: implement actual model inference with tf.tidy
    return {
      stems: new Array(this.numStems)
        .fill(null)
        .map(() => new Float32Array(chunkPCM.length)),
    };
  }
}
