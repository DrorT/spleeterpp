// Inference adapter stub for TF.js models

export class InferenceEngine {
  constructor(opts = {}) {
    this.numStems = opts.numStems || 2;
    this.model = null;
  }

  async load(modelUrl) {
    // TODO: integrate with TF.js (tf.loadGraphModel)
    this.model = { url: modelUrl };
  }

  isLoaded() {
    return !!this.model;
  }

  async runChunk(chunkPCM /* Float32Array */, sampleRate) {
    // TODO: implement actual model inference
    return {
      stems: new Array(this.numStems)
        .fill(null)
        .map(() => new Float32Array(chunkPCM.length)),
    };
  }
}
