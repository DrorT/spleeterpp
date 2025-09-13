// Inference adapter stub for TF.js models
import { ensureTF } from './tf-backend.js';

export class InferenceEngine {
  constructor(opts = {}) {
    this.numStems = opts.numStems || 2;
    this.model = null;
  }

  async load(modelUrl) {
    const tf = await ensureTF();
    // TODO: integrate with tf.loadGraphModel and proper input/output mapping
    this.model = { url: modelUrl, tfVersion: tf?.version?.tfjs };
  }

  isLoaded() {
    return !!this.model;
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
