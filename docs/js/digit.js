/* ============================================================
   digit.js — draw a digit, watch both learning rules argue.

   The fiddly part is not the network, it's the preprocessing.
   MNIST digits are not just "a 28x28 picture of a number": each
   one was size-normalised to fit a 20x20 box and then shifted so
   its CENTRE OF MASS sits at the centre of a 28x28 field. Feed a
   naively downscaled drawing to the model and it will be wrong
   for reasons that have nothing to do with the learning rule.

   So we do exactly what MNIST did.
   ============================================================ */

const PAD_N = 280;          // internal drawing resolution
let padCtx = null;
let drawing = false;
let inked = false;
let lastPt = null;

function padSetup() {
  const pad = document.getElementById("pad");
  if (!pad) return;
  pad.width = PAD_N;
  pad.height = PAD_N;
  padCtx = pad.getContext("2d", { willReadFrequently: true });
  padClear();

  const pos = (e) => {
    const r = pad.getBoundingClientRect();
    const t = e.touches ? e.touches[0] : e;
    return [
      ((t.clientX - r.left) / r.width) * PAD_N,
      ((t.clientY - r.top) / r.height) * PAD_N,
    ];
  };

  const start = (e) => { e.preventDefault(); drawing = true; lastPt = pos(e); dot(lastPt); };
  const move  = (e) => {
    if (!drawing) return;
    e.preventDefault();
    const p = pos(e);
    stroke(lastPt, p);
    lastPt = p;
    classify();
  };
  const end   = () => { if (drawing) { drawing = false; lastPt = null; classify(); } };

  pad.addEventListener("pointerdown", start);
  pad.addEventListener("pointermove", move);
  window.addEventListener("pointerup", end);
  pad.addEventListener("pointerleave", end);

  document.getElementById("pad-clear").addEventListener("click", () => {
    padClear();
    render(null);
  });
  document.getElementById("pad-example").addEventListener("click", drawRandomExample);
}

function padClear() {
  padCtx.fillStyle = "#000";
  padCtx.fillRect(0, 0, PAD_N, PAD_N);
  inked = false;
}

function dot(p) {
  padCtx.fillStyle = "#fff";
  padCtx.beginPath();
  padCtx.arc(p[0], p[1], 11, 0, Math.PI * 2);
  padCtx.fill();
  inked = true;
}
function stroke(a, b) {
  padCtx.strokeStyle = "#fff";
  padCtx.lineWidth = 22;
  padCtx.lineCap = "round";
  padCtx.lineJoin = "round";
  padCtx.beginPath();
  padCtx.moveTo(a[0], a[1]);
  padCtx.lineTo(b[0], b[1]);
  padCtx.stroke();
  inked = true;
}

/* ---- MNIST's own recipe ---- */
function padTo784() {
  const img = padCtx.getImageData(0, 0, PAD_N, PAD_N).data;

  /* 1. bounding box of the ink */
  let x0 = PAD_N, y0 = PAD_N, x1 = -1, y1 = -1;
  for (let y = 0; y < PAD_N; y++) {
    for (let x = 0; x < PAD_N; x++) {
      if (img[(y * PAD_N + x) * 4] > 24) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  if (x1 < 0) return null;

  /* 2. scale the longer side to 20px, keeping the aspect ratio */
  const w = x1 - x0 + 1, h = y1 - y0 + 1;
  const s = 20 / Math.max(w, h);
  const nw = Math.max(1, Math.round(w * s));
  const nh = Math.max(1, Math.round(h * s));

  const tmp = document.createElement("canvas");
  tmp.width = 28; tmp.height = 28;
  const t = tmp.getContext("2d", { willReadFrequently: true });
  t.fillStyle = "#000";
  t.fillRect(0, 0, 28, 28);
  t.imageSmoothingEnabled = true;
  t.imageSmoothingQuality = "high";
  t.drawImage(
    document.getElementById("pad"),
    x0, y0, w, h,
    Math.round((28 - nw) / 2), Math.round((28 - nh) / 2), nw, nh
  );

  /* 3. shift so the centre of mass lands at the centre of the field */
  const d = t.getImageData(0, 0, 28, 28).data;
  let m = 0, mx = 0, my = 0;
  const g = new Float32Array(784);
  for (let i = 0; i < 784; i++) {
    const v = d[i * 4] / 255;
    g[i] = v;
    m += v;
    mx += (i % 28) * v;
    my += Math.floor(i / 28) * v;
  }
  if (m < 1e-6) return null;
  const dx = Math.round(13.5 - mx / m);
  const dy = Math.round(13.5 - my / m);

  const out = new Float32Array(784);
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const sx = x - dx, sy = y - dy;
      if (sx >= 0 && sx < 28 && sy >= 0 && sy < 28) out[y * 28 + x] = g[sy * 28 + sx];
    }
  }
  return out;
}

/* ---- run both models ----
   Each classify() is roughly a million float multiplies (two 400x784
   matrices plus the Hebbian settling steps). That is fine once, and not
   fine sixty times a second, so it's clamped to one run per frame. */
let queued = false;
function classify() {
  if (!NET.ready || queued) return;
  queued = true;
  requestAnimationFrame(() => {
    queued = false;
    const px = inked ? padTo784() : null;
    if (!px) { render(null); return; }
    const x = preprocess(px);
    render({
      backprop: FORWARD.backprop(x),
      hebbian:  FORWARD.hebbian(x),
    });
  });
}

function render(res) {
  ["backprop", "hebbian"].forEach((k) => {
    const row = document.querySelector(`.verdict[data-m="${k}"]`);
    if (!row) return;
    const guess = row.querySelector(".guess");
    const bars = row.querySelectorAll(".bars10 i");
    const hid = row.querySelectorAll(".hidden-strip i");

    if (!res) {
      guess.textContent = "–";
      guess.classList.add("none");
      bars.forEach((b) => { b.style.height = "1px"; b.classList.remove("top"); });
      hid.forEach((h) => { h.style.opacity = 0.12; });
      return;
    }

    const p = res[k].probs;
    let best = 0;
    for (let i = 1; i < 10; i++) if (p[i] > p[best]) best = i;
    guess.textContent = best;
    guess.classList.remove("none");

    bars.forEach((b, i) => {
      b.style.height = Math.max(1, p[i] * 34) + "px";
      b.classList.toggle("top", i === best);
    });

    /* the hidden layer, coarse-grained: 400 units -> 40 blocks */
    const hvec = res[k].hidden;
    let mx = 1e-8;
    for (const v of hvec) if (v > mx) mx = v;
    hid.forEach((cell, i) => {
      let s = 0;
      for (let j = i * 10; j < (i + 1) * 10; j++) s += hvec[j] || 0;
      cell.style.opacity = (0.10 + 0.9 * Math.min(1, (s / 10) / mx)).toFixed(3);
    });
  });
}

/* ---- a real test digit, straight from the dataset ---- */
function drawRandomExample() {
  if (!NET.ready || !NET.examples.length) return;
  const e = NET.examples[(Math.random() * NET.examples.length) | 0];

  const tmp = document.createElement("canvas");
  tmp.width = 28; tmp.height = 28;
  const t = tmp.getContext("2d");
  const im = t.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const v = e.px[i];
    im.data[i * 4] = v; im.data[i * 4 + 1] = v; im.data[i * 4 + 2] = v; im.data[i * 4 + 3] = 255;
  }
  t.putImageData(im, 0, 0);

  padCtx.fillStyle = "#000";
  padCtx.fillRect(0, 0, PAD_N, PAD_N);
  padCtx.imageSmoothingEnabled = true;
  padCtx.imageSmoothingQuality = "high";
  padCtx.drawImage(tmp, 0, 0, PAD_N, PAD_N);
  inked = true;

  const truth = document.getElementById("pad-truth");
  if (truth) truth.textContent = "actually a " + e.y;
  classify();
}

/* ---- the honest empty state ---- */
function digitStatus() {
  const box = document.getElementById("demo-status");
  if (!box) return;
  if (NET.ready) {
    const bp = NET.models.backprop.acc, hb = NET.models.hebbian.acc;
    box.innerHTML =
      `<b>Live.</b> These are real trained weights running in your browser — ` +
      `no server, no API call. This exact backprop weight set scores ` +
      `<b>${bp.toFixed(2)}%</b> on the test set; this exact Hebbian one scores ` +
      `<b>${hb.toFixed(2)}%</b>. Draw badly on purpose. That's when it gets interesting.`;
    box.style.borderStyle = "solid";
  } else {
    box.innerHTML =
      `<b>Weights not loaded.</b> ${NET.error || ""}<br><br>` +
      `Run <code>python models/export_web_assets.py</code> from the repo root to write ` +
      `<code>assets/models.json</code>, then serve the folder with ` +
      `<code>python -m http.server</code>. Opening index.html straight off the disk ` +
      `will not work — browsers block <code>fetch()</code> on <code>file://</code>.`;
  }
}

async function initDigit() {
  padSetup();
  await loadNet();
  digitStatus();
  initFilters();
  render(null);
}
