/* ============================================================
   net.js — runs the trained networks in the browser.

   Loads assets/models.json (written by models/export_web_assets.py).
   Matrices are int8-quantised with a per-matrix scale, so the
   whole thing is about a megabyte instead of ten.

   The preprocessing here MUST match the preprocessing in the
   export script exactly, or the demo will quietly give garbage.
   Both do: /255  ->  subtract the dataset mean image  ->  L2
   normalise to unit length. The spec is carried in the JSON so
   the two can never drift apart.
   ============================================================ */

const NET = {
  ready: false,
  error: null,
  meta: null,
  pre: null,
  models: {},
  examples: [],
};

/* ---------- decode ---------- */
function b64ToI8(b64) {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Int8Array(u8.buffer);
}
function b64ToU8(b64) {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8;
}

/* A quantised matrix -> Float32Array, row-major [rows x cols]. */
function deqMat(m) {
  const q = b64ToI8(m.data);
  const out = new Float32Array(q.length);
  const s = m.scale;
  for (let i = 0; i < q.length; i++) out[i] = q[i] * s;
  return { rows: m.shape[0], cols: m.shape[1], v: out };
}

/* y = M x + b   (M is [rows x cols], x is length cols) */
function matVec(M, x, b) {
  const { rows, cols, v } = M;
  const y = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let s = b ? b[r] : 0;
    const off = r * cols;
    for (let c = 0; c < cols; c++) s += v[off + c] * x[c];
    y[r] = s;
  }
  return y;
}

function relu(v) {
  const o = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) o[i] = v[i] > 0 ? v[i] : 0;
  return o;
}

function softmax(v) {
  let m = -Infinity;
  for (const x of v) if (x > m) m = x;
  let s = 0;
  const o = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) { o[i] = Math.exp(v[i] - m); s += o[i]; }
  for (let i = 0; i < o.length; i++) o[i] /= s;
  return o;
}

/* ---------- preprocessing: identical to the export script ---------- */
function preprocess(px /* Float32Array(784), values 0..1 */) {
  const x = new Float32Array(784);
  const mu = NET.pre.mean_image;
  for (let i = 0; i < 784; i++) x[i] = px[i] - mu[i];
  if (NET.pre.l2_normalize) {
    let n = 0;
    for (let i = 0; i < 784; i++) n += x[i] * x[i];
    n = Math.sqrt(n) + 1e-8;
    for (let i = 0; i < 784; i++) x[i] /= n;
  }
  return x;
}

/* ---------- forward passes ---------- */

/* Backprop MLP: h = relu(W1 x + b1);  logits = W2 h + b2 */
function forwardBackprop(x) {
  const m = NET.models.backprop;
  const h = relu(matVec(m.W1, x, m.b1));
  const logits = matVec(m.W2, h, m.b2);
  return { hidden: h, probs: softmax(logits) };
}

/* Hebbian layer: activity settles under mutual inhibition,
   then a supervised linear readout sits on top.
     a = W1 x
     y <- relu(a - theta - L yᵀ)      (a few fixed-point steps)
     logits = Wr y + br                                        */
function forwardHebbian(x) {
  const m = NET.models.hebbian;
  const a = matVec(m.W1, x, null);
  const n = a.length;
  let y = new Float32Array(n);

  const steps = m.L ? (m.settle_steps || 3) : 1;
  for (let s = 0; s < steps; s++) {
    const inh = m.L ? matVec(m.L, y, null) : null;
    const ny = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const v = a[i] - m.theta[i] - (inh ? inh[i] : 0);
      ny[i] = v > 0 ? v : 0;
    }
    y = ny;
  }
  const logits = matVec(m.Wr, y, m.br);
  return { hidden: y, probs: softmax(logits) };
}

const FORWARD = { backprop: forwardBackprop, hebbian: forwardHebbian };

/* ---------- load ---------- */
async function loadNet() {
  try {
    const res = await fetch("assets/models.json", { cache: "no-store" });
    if (!res.ok) throw new Error("assets/models.json not found (HTTP " + res.status + ")");
    const j = await res.json();

    NET.meta = j.meta;
    NET.pre = { mean_image: Float32Array.from(j.pre.mean_image), l2_normalize: !!j.pre.l2_normalize };

    const bp = j.models.backprop;
    NET.models.backprop = {
      acc: bp.acc,
      W1: deqMat(bp.W1), b1: Float32Array.from(bp.b1),
      W2: deqMat(bp.W2), b2: Float32Array.from(bp.b2),
    };

    const hb = j.models.hebbian;
    NET.models.hebbian = {
      acc: hb.acc,
      W1: deqMat(hb.W1),
      theta: Float32Array.from(hb.theta),
      L: hb.L ? deqMat(hb.L) : null,
      settle_steps: hb.settle_steps || 3,
      Wr: deqMat(hb.Wr), br: Float32Array.from(hb.br),
    };

    NET.examples = (j.examples || []).map((e) => ({
      px: b64ToU8(e.px), y: e.y,
    }));

    NET.ready = true;
  } catch (err) {
    NET.error = err.message;
  }
  return NET;
}
