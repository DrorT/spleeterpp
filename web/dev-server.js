// Minimal Express server that serves /web with COOP/COEP headers
// Enables cross-origin isolation so TF.js WASM can use threads when supported.
const express = require("express");
const path = require("path");
const fs = require("fs");
const https = require("https");

const app = express();

// COOP/COEP headers for all resources
app.use((req, res, next) => {
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  next();
});

// Serve static files from current directory
const root = __dirname;
app.use(express.static(root, { fallthrough: true }));

// Mirror TFJS WASM binaries locally under /tfjs-wasm
const wasmDir = path.join(root, "tfjs-wasm");
const wasmFiles = [
  // File names for 4.18.0; tfjs will choose wasm/wasm-simd/wasm-threaded based on support
  "tfjs-backend-wasm.wasm",
  "tfjs-backend-wasm-simd.wasm",
  "tfjs-backend-wasm-threaded-simd.wasm",
];
function ensureDir(p) {
  if (!fs.existsSync(p)) fs.mkdirSync(p, { recursive: true });
}
function download(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    https
      .get(url, (res) => {
        if (res.statusCode !== 200) {
          return reject(new Error(`GET ${url} -> ${res.statusCode}`));
        }
        res.pipe(file);
        file.on("finish", () => file.close(resolve));
      })
      .on("error", (err) => {
        fs.unlink(dest, () => reject(err));
      });
  });
}
async function prepareWasm() {
  try {
    ensureDir(wasmDir);
    const base =
      "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.18.0/dist/";
    for (const f of wasmFiles) {
      const out = path.join(wasmDir, f);
      if (!fs.existsSync(out)) {
        console.log(`Downloading ${f} ...`);
        await download(base + f, out);
      }
    }
  } catch (e) {
    console.warn("Could not prepare TFJS WASM files:", e.message);
  }
}
// Serve local copies at /tfjs-wasm with correct content-type
app.use("/tfjs-wasm", (req, res, next) => {
  if (req.path.endsWith(".wasm")) {
    res.type("application/wasm");
  }
  next();
});
app.use("/tfjs-wasm", express.static(wasmDir, { fallthrough: true }));

// For convenience, map / to index.html
app.get("/", (req, res) => {
  res.sendFile(path.join(root, "index.html"));
});

const port = process.env.PORT || 8080;
prepareWasm().then(() => {
  app.listen(port, () => {
    console.log(`WASM-COI dev server running at http://localhost:${port}`);
    console.log(`Serving TFJS WASM from /tfjs-wasm`);
  });
});
