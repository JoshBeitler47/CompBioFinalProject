/* ============================================================
   filters.js — what does a hidden unit actually look for?

   Each hidden unit has 784 incoming weights: one per pixel.
   So its weights ARE a 28x28 image. Draw them back out and you
   can see, directly, what the layer decided was worth noticing.

   Colour is signed: the unit's own colour where the weight is
   POSITIVE (this pixel excites me), rose where it is NEGATIVE
   (this pixel argues against me). Each filter is normalised by
   its own peak, so we're looking at shape, not amplitude.

   The random filters aren't shipped in the asset file — they're
   generated right here from a seeded RNG, which is exactly as
   valid as shipping a megabyte of noise.
   ============================================================ */

const HEX = (h) => [
  parseInt(h.slice(1, 3), 16),
  parseInt(h.slice(3, 5), 16),
  parseInt(h.slice(5, 7), 16),
];
const NEG = HEX(PALETTE.flag);
const BG = HEX("#0D1418");
const N_SHOW = 72;

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* Draw one 784-length weight vector into a 28x28 canvas. */
function paintFilter(cv, w, offset, posHex) {
  const POS = HEX(posHex);
  const ctx = cv.getContext("2d");
  const im = ctx.createImageData(28, 28);

  let mx = 1e-8;
  for (let i = 0; i < 784; i++) {
    const a = Math.abs(w[offset + i]);
    if (a > mx) mx = a;
  }

  for (let i = 0; i < 784; i++) {
    const v = w[offset + i] / mx;          // -1 .. 1
    const t = Math.min(1, Math.abs(v));
    const C = v >= 0 ? POS : NEG;
    im.data[i * 4]     = BG[0] + (C[0] - BG[0]) * t;
    im.data[i * 4 + 1] = BG[1] + (C[1] - BG[1]) * t;
    im.data[i * 4 + 2] = BG[2] + (C[2] - BG[2]) * t;
    im.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
}

function filterSet(which) {
  if (which === "random") {
    const rnd = mulberry32(20260713);
    const v = new Float32Array(N_SHOW * 784);
    for (let i = 0; i < v.length; i++) {
      /* sum of uniforms ~ gaussian, which is how the layer was initialised */
      v[i] = (rnd() + rnd() + rnd() + rnd() - 2) * 0.5;
    }
    return { v, rows: N_SHOW, cols: 784, stride: 1, color: PALETTE.random };
  }
  if (!NET.ready) return null;
  const M = NET.models[which].W1;                 // [400 x 784]
  const stride = Math.max(1, Math.floor(M.rows / N_SHOW));
  return {
    v: M.v, rows: M.rows, cols: 784, stride,
    color: which === "hebbian" ? PALETTE.hebbian : PALETTE.backprop,
  };
}

const FILTER_BLURB = {
  hebbian:
    "Recognisable strokes and whole-digit templates. No label was ever shown to this layer — " +
    "it found these by watching pixels co-occur, and nothing else. This is Hubel & Wiesel's " +
    "point, on our bench: local, unsupervised learning on real input produces feature detectors.",
  backprop:
    "Speckle. Backprop's hidden units are not trying to look like digits — they're trying to " +
    "be <em>useful to the layer above them</em>, and a highly effective feature can look like " +
    "nothing at all. It wins by 8.7 points while being far less interpretable.",
  random:
    "The control: pure noise, never trained. It still reaches 85.66% once a supervised readout " +
    "is bolted on, because 400 random projections of a digit are already a decent description " +
    "of it. This is the bar the Hebbian layer actually has to clear.",
};

function initFilters() {
  const grid = document.getElementById("filters-grid");
  const seg = document.getElementById("filters-seg");
  const blurb = document.getElementById("filters-blurb");
  if (!grid) return;

  function show(which) {
    seg.querySelectorAll("button").forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.f === which))
    );
    blurb.innerHTML = FILTER_BLURB[which];

    const set = filterSet(which);
    grid.innerHTML = "";

    if (!set) {
      grid.innerHTML =
        `<div class="filters-empty" style="grid-column:1/-1">` +
        `Weights not loaded — run <code>models/export_web_assets.py</code>. ` +
        `The random control below needs no weights, so try that one.` +
        `</div>`;
      return;
    }

    const n = Math.min(N_SHOW, Math.floor(set.rows / set.stride));
    for (let i = 0; i < n; i++) {
      const cv = document.createElement("canvas");
      cv.width = 28; cv.height = 28;
      const unit = i * set.stride;
      cv.title = `hidden unit ${unit}`;
      paintFilter(cv, set.v, unit * set.cols, set.color);
      grid.appendChild(cv);
    }
  }

  seg.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (b) show(b.dataset.f);
  });

  show(NET.ready ? "hebbian" : "random");
}
