/* ============================================================
   charts.js — every chart on this page is hand-built SVG,
   drawn from data.js. No chart library, no image files.
   Change a number in data.js and the chart redraws.
   ============================================================ */

const NS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, text) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (text != null) n.textContent = text;
  return n;
}

/* Shared hatch pattern for "we haven't run this yet" bars. */
function defsHatch(svg) {
  const defs = el("defs");
  const p = el("pattern", {
    id: "hatch", width: 6, height: 6,
    patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)",
  });
  p.appendChild(el("rect", { width: 6, height: 6, fill: "transparent" }));
  p.appendChild(el("line", { x1: 0, y1: 0, x2: 0, y2: 6, stroke: PALETTE.mute, "stroke-width": 2 }));
  defs.appendChild(p);
  svg.appendChild(defs);
}

/* ------------------------------------------------------------
   THE LADDER
   Horizontal bars, one per model. Axis is truncated at 70% and
   says so — the whole point is the size of the gaps, and a
   0-100 axis would hide them. Every bar also carries its own
   number, so nothing depends on reading the axis.
   ------------------------------------------------------------ */
function drawLadder(mount, which, onFocus) {
  const set = LADDER[which];
  mount.innerHTML = "";

  const W = 720, X0 = 128, X1 = 660;
  const BAR = 34, GAP = 15, TOP = 26;
  const H = TOP + set.rows.length * (BAR + GAP) + 34;
  const LO = 70, HI = 100;
  const sx = (v) => X0 + ((v - LO) / (HI - LO)) * (X1 - X0);

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": `Test accuracy on ${set.label} for four hidden-layer learning rules.` });
  defsHatch(svg);

  /* gridlines */
  for (let v = 70; v <= 100; v += 5) {
    svg.appendChild(el("line", {
      x1: sx(v), y1: TOP - 12, x2: sx(v), y2: H - 30,
      stroke: v === 70 ? "rgba(233,240,242,0.22)" : "rgba(233,240,242,0.07)",
      "stroke-width": 1,
    }));
    svg.appendChild(el("text", {
      x: sx(v), y: H - 14, fill: PALETTE.mute, "text-anchor": "middle", "font-size": 10, "letter-spacing": "0.05em",
    }, v + "%"));
  }

  /* the axis-break glyph — do not hide a truncated axis */
  svg.appendChild(el("path", {
    d: `M ${X0 - 7} ${H - 30} l 4 -5 l -4 -5`,
    stroke: "rgba(233,240,242,0.35)", fill: "none", "stroke-width": 1,
  }));

  set.rows.forEach((r, i) => {
    const y = TOP + i * (BAR + GAP);
    const g = el("g", {
      class: "bar-row", tabindex: 0, role: "button",
      "aria-label": r.pending
        ? `${r.name}: not yet run.`
        : `${r.name}: ${r.mean.toFixed(2)} percent, plus or minus ${r.std.toFixed(2)}.`,
    });
    g.style.cursor = "pointer";

    /* name */
    g.appendChild(el("text", {
      x: X0 - 16, y: y + BAR / 2 + 4, fill: PALETTE.ink, "text-anchor": "end", "font-size": 12, "font-weight": 500,
      "letter-spacing": "-0.01em",
    }, r.name));

    /* hit target */
    g.appendChild(el("rect", {
      x: 0, y: y - GAP / 2, width: W, height: BAR + GAP, fill: "transparent",
    }));

    if (r.pending) {
      const w = sx(97) - sx(LO);
      g.appendChild(el("rect", {
        x: sx(LO), y, width: w, height: BAR, fill: "url(#hatch)",
        stroke: "rgba(233,240,242,0.18)", "stroke-dasharray": "3 3", rx: 1,
      }));
      g.appendChild(el("text", {
        x: sx(LO) + 12, y: y + BAR / 2 + 4, fill: PALETTE.mute, "font-size": 11, "letter-spacing": "0.05em",
      }, "NOT YET RUN — see data.js"));
    } else {
      const w = Math.max(2, sx(r.mean) - sx(LO));
      const bar = el("rect", {
        x: sx(LO), y, width: 0, height: BAR, fill: r.color, rx: 1, opacity: 0.92,
      });
      bar.dataset.w = w;
      g.appendChild(bar);

      /* error bar: mean ± std */
      if (r.std > 0) {
        const a = sx(r.mean - r.std), b = sx(r.mean + r.std), cy = y + BAR / 2;
        const wh = el("g", { class: "whisk", opacity: 0 });
        wh.appendChild(el("line", { x1: a, y1: cy, x2: b, y2: cy, stroke: "#0D1418", "stroke-width": 1.5 }));
        wh.appendChild(el("line", { x1: a, y1: cy - 5, x2: a, y2: cy + 5, stroke: "#0D1418", "stroke-width": 1.5 }));
        wh.appendChild(el("line", { x1: b, y1: cy - 5, x2: b, y2: cy + 5, stroke: "#0D1418", "stroke-width": 1.5 }));
        g.appendChild(wh);
      }

      const t = el("text", {
        x: sx(r.mean) + 10, y: y + BAR / 2 + 4, fill: r.color, opacity: 0, "font-size": 12.5, "font-weight": 600,
        "letter-spacing": "-0.02em",
      }, r.std ? `${r.mean.toFixed(2)}%  ±${r.std.toFixed(2)}` : `${r.mean.toFixed(2)}%`);
      t.classList.add("val");
      g.appendChild(t);
    }

    const fire = () => onFocus && onFocus(r.key);
    g.addEventListener("mouseenter", fire);
    g.addEventListener("focus", fire);
    g.addEventListener("click", fire);
    g.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fire(); }
    });

    svg.appendChild(g);
  });

  mount.appendChild(svg);

  /* grow bars in — the ladder should be *climbed*, not just appear */
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const bars = svg.querySelectorAll("rect[data-w]");
  const vals = svg.querySelectorAll("text.val");
  const whisks = svg.querySelectorAll("g.whisk");
  bars.forEach((b, i) => {
    const w = +b.dataset.w;
    if (reduce) { b.setAttribute("width", w); return; }
    b.style.transition = `width 0.75s cubic-bezier(0.2,0.75,0.3,1) ${i * 110}ms`;
    requestAnimationFrame(() => requestAnimationFrame(() => b.setAttribute("width", w)));
  });
  vals.forEach((t, i) => {
    if (reduce) { t.setAttribute("opacity", 1); return; }
    t.style.transition = `opacity 0.4s ease ${500 + i * 110}ms`;
    requestAnimationFrame(() => requestAnimationFrame(() => t.setAttribute("opacity", 1)));
  });
  whisks.forEach((t, i) => {
    if (reduce) { t.setAttribute("opacity", 0.55); return; }
    t.style.transition = `opacity 0.4s ease ${560 + i * 110}ms`;
    requestAnimationFrame(() => requestAnimationFrame(() => t.setAttribute("opacity", 0.55)));
  });
}

/* ------------------------------------------------------------
   CATASTROPHIC FORGETTING
   Three bars per condition: what it knew, what it still knows,
   and what a fresh readout can recover from the same features.
   The third bar is the whole finding.
   ------------------------------------------------------------ */
function drawForgetting(mount) {
  mount.innerHTML = "";
  const rows = FORGETTING.rows;

  const W = 720, H = 330;
  const X0 = 46, X1 = 700, Y0 = 24, Y1 = 258;
  const sy = (v) => Y1 - (v / 100) * (Y1 - Y0);

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Split-MNIST forgetting. Every model loses nearly all its accuracy on digits 0 to 4 after learning 5 to 9, but a freshly trained readout recovers almost all of it from the same features." });

  for (let v = 0; v <= 100; v += 25) {
    svg.appendChild(el("line", {
      x1: X0, y1: sy(v), x2: X1, y2: sy(v),
      stroke: v === 0 ? "rgba(233,240,242,0.22)" : "rgba(233,240,242,0.07)",
    }));
    svg.appendChild(el("text", {
      x: X0 - 10, y: sy(v) + 4, fill: PALETTE.mute, "text-anchor": "end", "font-size": 10,
    }, v));
  }

  const slot = (X1 - X0) / rows.length;
  const BW = 21, PAD = 4;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const grown = [];

  rows.forEach((r, i) => {
    const cx = X0 + slot * i + slot / 2;
    const x0 = cx - (BW * 1.5 + PAD);

    const spec = [
      { v: r.before, fill: r.color,      op: 0.32, lab: "before" },
      { v: r.after,  fill: PALETTE.flag, op: 1.00, lab: "after"  },
      { v: r.fresh,  fill: r.color,      op: 0.95, lab: "fresh"  },
    ];

    spec.forEach((s, j) => {
      const x = x0 + j * (BW + PAD);
      const h = Math.max(1.5, (s.v / 100) * (Y1 - Y0));
      const rect = el("rect", {
        x, y: Y1 - 0, width: BW, height: 0,
        fill: s.fill, opacity: s.op, rx: 1,
      });
      grown.push([rect, Y1 - h, h]);
      svg.appendChild(rect);

      const g2 = el("g");
      g2.appendChild(el("rect", { x, y: Y0, width: BW, height: Y1 - Y0, fill: "transparent" }));
      const tip = el("text", {
        x: x + BW / 2, y: Y1 - h - 7, fill: s.fill, "text-anchor": "middle", opacity: 0, "font-size": 9.5, "font-weight": 600,
      }, s.v.toFixed(1));
      g2.appendChild(tip);
      g2.addEventListener("mouseenter", () => tip.setAttribute("opacity", 1));
      g2.addEventListener("mouseleave", () => tip.setAttribute("opacity", 0));
      svg.appendChild(g2);
    });

    /* condition label, two lines so the (sparse) variants read cleanly */
    const parts = r.name.replace(")", "").split(" (");
    svg.appendChild(el("text", {
      x: cx, y: Y1 + 20, fill: r.sparse ? PALETTE.dim : PALETTE.ink, "text-anchor": "middle", "font-size": 10.5, "font-weight": 500,
    }, parts[0]));
    if (parts[1]) {
      svg.appendChild(el("text", {
        x: cx, y: Y1 + 33, fill: PALETTE.mute, "text-anchor": "middle", "font-size": 9, "letter-spacing": "0.06em",
      }, parts[1].toUpperCase()));
    }

    /* the loss, called out in the flag colour */
    svg.appendChild(el("text", {
      x: cx, y: Y1 + 52, fill: PALETTE.flag, "text-anchor": "middle", "font-size": 9.5, "font-weight": 600,
    }, `−${r.lost.toFixed(1)} pts`));
  });

  svg.appendChild(el("text", {
    x: X0 - 10, y: Y0 - 8, fill: PALETTE.mute, "text-anchor": "start", "font-size": 9, "letter-spacing": "0.1em",
  }, "ACCURACY ON DIGITS 0–4 (%)"));

  mount.appendChild(svg);

  grown.forEach(([rect, y, h], i) => {
    if (reduce) { rect.setAttribute("y", y); rect.setAttribute("height", h); return; }
    rect.style.transition = `height 0.6s cubic-bezier(0.2,0.75,0.3,1) ${i * 35}ms, y 0.6s cubic-bezier(0.2,0.75,0.3,1) ${i * 35}ms`;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      rect.setAttribute("y", y); rect.setAttribute("height", h);
    }));
  });
}

/* ------------------------------------------------------------
   THE TUNING SLOPE
   One knob, two datasets, opposite signs. That's the finding.
   ------------------------------------------------------------ */
function drawTuning(mount) {
  mount.innerHTML = "";
  const W = 480, H = 250;
  const X0 = 118, X1 = 348, Y0 = 30, Y1 = 196;
  const LO = 70, HI = 92;
  const sy = (v) => Y1 - ((v - LO) / (HI - LO)) * (Y1 - Y0);

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "The same Hebbian sparsity setting raises MNIST accuracy slightly and lowers Fashion-MNIST accuracy." });

  [["DEFAULT", X0], ["TUNED", X1]].forEach(([t, x]) => {
    svg.appendChild(el("line", { x1: x, y1: Y0 - 6, x2: x, y2: Y1 + 6, stroke: "rgba(233,240,242,0.10)" }));
    svg.appendChild(el("text", {
      x, y: Y1 + 26, fill: PALETTE.mute, "text-anchor": "middle", "font-size": 9.5, "letter-spacing": "0.14em",
    }, t));
  });

  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  TUNING.forEach((t) => {
    const up = t.tuned >= t.base;
    const c = up ? PALETTE.hebbian : PALETTE.flag;
    const y0 = sy(t.base), y1 = sy(t.tuned);

    const line = el("line", { x1: X0, y1: y0, x2: X0, y2: y0, stroke: c, "stroke-width": 2 });
    svg.appendChild(line);
    svg.appendChild(el("circle", { cx: X0, cy: y0, r: 4, fill: c }));
    const dot2 = el("circle", { cx: X0, cy: y0, r: 4, fill: c });
    svg.appendChild(dot2);

    svg.appendChild(el("text", {
      x: X0 - 14, y: y0 + 4, fill: PALETTE.ink, "text-anchor": "end", "font-size": 11, "font-weight": 500,
    }, t.task));
    svg.appendChild(el("text", {
      x: X0 - 14, y: y0 + 17, fill: PALETTE.mute, "text-anchor": "end", "font-size": 9.5,
    }, t.base.toFixed(2) + "%"));

    svg.appendChild(el("text", {
      x: X1 + 14, y: y1 + 4, fill: c, "text-anchor": "start", "font-size": 11.5, "font-weight": 600,
    }, t.tuned.toFixed(2) + "%"));
    svg.appendChild(el("text", {
      x: X1 + 14, y: y1 + 17, fill: c, "text-anchor": "start", "font-size": 9.5, opacity: 0.75,
    }, (up ? "+" : "−") + Math.abs(t.tuned - t.base).toFixed(2) + " pts"));

    if (reduce) {
      line.setAttribute("x2", X1); line.setAttribute("y2", y1);
      dot2.setAttribute("cx", X1); dot2.setAttribute("cy", y1);
    } else {
      line.style.transition = "all 0.8s cubic-bezier(0.2,0.75,0.3,1) 200ms";
      dot2.style.transition = "all 0.8s cubic-bezier(0.2,0.75,0.3,1) 200ms";
      requestAnimationFrame(() => requestAnimationFrame(() => {
        line.setAttribute("x2", X1); line.setAttribute("y2", y1);
        dot2.setAttribute("cx", X1); dot2.setAttribute("cy", y1);
      }));
    }
  });

  mount.appendChild(svg);
}
