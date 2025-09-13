import { ensureTF } from './tf-backend.js';

export async function loadGraphModelWithCache(url, opts = {}) {
  const tf = await ensureTF();
  const loadOpts = { fromTFHub: false, ...opts };
  // Enable IndexedDB caching implicitly by using HTTP(S) URL; tfjs handles caching by default for GraphModel
  const model = await tf.loadGraphModel(url, loadOpts);
  // Touch the model once to ensure weights are cached
  if (opts.warmup) {
    const dummy = tf.zeros([1, 1024, 2]); // placeholder shape; adjust later once model IO is known
    const out = model.execute({ input_audio: dummy });
    if (out && typeof out.dispose === 'function') out.dispose();
    dummy.dispose();
  }
  return model;
}
