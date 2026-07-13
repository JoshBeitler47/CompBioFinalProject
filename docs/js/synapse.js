/* ============================================================
   synapse.js — THE SIGNATURE.

   One synapse is highlighted in gold. For each learning rule
   we black out everything that synapse cannot physically read.

   Hebbian : three things stay lit. That's all a synapse has.
   Backprop: a wire appears carrying a copy of W2 backwards.
             That wire does not exist in a brain.
   Feedback: the same wire, but carrying a fixed random matrix.
     alignment  Nothing has to be copied. It still learns.

   The whole argument of the project is this one widget.
   ============================================================ */

const RULES = {
  hebbian: {
    math: [
      ['<span class="tk-dim">Δw</span> = <span class="tk-dim">η</span> · ', ''],
      ['<span class="tk-bio">y</span> · ( <span class="tk-bio">x</span> − <span class="tk-bio">y</span>·<span class="tk-bio">w</span> )', ''],
    ],
    line: '<span class="tk-dim">Δw</span> = <span class="tk-dim">η</span> · <span class="tk-bio">y</span> · ( <span class="tk-bio">x</span> − <span class="tk-bio">y</span> · <span class="tk-bio">w</span> )',
    needs: 'Needs <span class="tk-bio">x</span> (the cell before it), <span class="tk-bio">y</span> (the cell after it), and <span class="tk-bio">w</span> (its own strength). Nothing else in the network exists as far as this synapse is concerned.',
  },
  backprop: {
    line: '<span class="tk-dim">Δw</span> = <span class="tk-dim">η</span> · <span class="tk-bio">x</span> · [ ( <span class="tk-mach">δ</span> <span class="tk-flag">W2ᵀ</span> ) ⊙ <span class="tk-bio">f′(a)</span> ]',
    needs: 'Needs <span class="tk-bio">x</span> and <span class="tk-bio">a</span> locally — fine. But it also needs <span class="tk-mach">δ</span>, an error computed at the far end of the network from the <em>correct label</em>, routed back through <span class="tk-flag">W2ᵀ</span>: the numerical values of 4,000 output weights this synapse has never touched.',
  },
  fa: {
    line: '<span class="tk-dim">Δw</span> = <span class="tk-dim">η</span> · <span class="tk-bio">x</span> · [ ( <span class="tk-mach">δ</span> <span class="tk-mid">B</span> ) ⊙ <span class="tk-bio">f′(a)</span> ]',
    needs: '<span class="tk-mid">B</span> is a random matrix, fixed at initialisation and <em>never updated</em>. Nothing is copied from anywhere. Lillicrap et al. showed this still trains — the forward weights simply learn to <em>align themselves</em> with the random feedback.',
  },
};

function buildScope(mount) {
  const IN  = [58, 108, 158, 208];
  const HID = [42, 92, 142, 192, 242];
  const OUT = [88, 138, 188];
  const XI = 66, XH = 234, XO = 402, XL = 512;
  const R = 11;

  const PRE = 1;   // the pre-synaptic cell   (input node index)
  const POST = 2;  // the post-synaptic cell  (hidden node index)

  const svg = el("svg", {
    id: "scope-svg", viewBox: "0 0 600 356",
    "data-mode": "hebbian", role: "img",
    "aria-label": "A network diagram in which everything one synapse cannot physically read is dimmed out.",
  });

  /* ---- forward wires: input -> hidden ---- */
  IN.forEach((y1, i) => HID.forEach((y2, j) => {
    const isThe = (i === PRE && j === POST);
    svg.appendChild(el("path", {
      d: `M ${XI + R} ${y1} L ${XH - R} ${y2}`,
      class: isThe ? "the-syn" : "wire dim",
      fill: "none",
    }));
  }));

  /* ---- forward wires: hidden -> output ---- */
  HID.forEach((y1) => OUT.forEach((y2) => {
    svg.appendChild(el("path", {
      d: `M ${XH + R} ${y1} L ${XO - R} ${y2}`,
      class: "wire w2 dim", fill: "none",
    }));
  }));

  /* ---- nodes ---- */
  const node = (x, y, cls, label) => {
    const g = el("g", { class: cls });
    g.appendChild(el("circle", { cx: x, cy: y, r: R, class: "n-body" }));
    if (label) g.appendChild(el("text", {
      x, y: y + 3, "text-anchor": "middle", class: "lab",
    }, label));
    return g;
  };

  IN.forEach((y, i) => svg.appendChild(
    node(XI, y, i === PRE ? "target" : "dim", i === PRE ? "x" : "")
  ));
  HID.forEach((y, j) => svg.appendChild(
    node(XH, y, j === POST ? "target" : "dim", j === POST ? "y" : "")
  ));
  OUT.forEach((y) => svg.appendChild(node(XO, y, "dim out-node")));

  /* ---- layer captions ---- */
  const cap = (x, t, s) => {
    svg.appendChild(el("text", { x, y: 302, "text-anchor": "middle", class: "layer-cap" }, t));
    svg.appendChild(el("text", { x, y: 313, "text-anchor": "middle", class: "layer-cap" }, s));
  };
  /* these are schematics: 4 dots is not 784 neurons, and the reader should know.
     drawn as circles rather than a "⋮" glyph, which not every font ships. */
  [[XI, 232], [XH, 266], [XO, 214]].forEach(([x, y]) =>
    [0, 6, 12].forEach((dy) =>
      svg.appendChild(el("circle", {
        cx: x, cy: y + dy, r: 1.3, fill: "#61757D", opacity: 0.7,
      }))
    )
  );

  cap(XI, "INPUT", "784 PIXELS");
  cap(XH, "HIDDEN", "400 UNITS");
  cap(XO, "OUTPUT", "10 DIGITS");

  /* the synapse under the microscope */
  svg.appendChild(el("text", {
    x: (XI + XH) / 2 - 4, y: 118, "text-anchor": "middle", class: "the-syn-lab",
  }, "w"));
  svg.appendChild(el("text", {
    x: (XI + XH) / 2 - 4, y: 22, "text-anchor": "middle", class: "the-syn-lab",
  }, "THE SYNAPSE"));
  svg.appendChild(el("path", {
    d: `M ${(XI + XH) / 2 - 4} 28 L ${(XI + XH) / 2 - 4} 108`,
    stroke: "rgba(232,179,60,0.35)", "stroke-width": 1, "stroke-dasharray": "2 3", fill: "none",
  }));

  /* ================= HEBBIAN ================= */
  const heb = el("g", { class: "heb-only" });
  heb.appendChild(el("text", {
    x: XO, y: 232, "text-anchor": "middle", class: "lab",
  }, "not involved"));
  heb.appendChild(el("text", {
    x: 300, y: 344, "text-anchor": "middle", class: "the-syn-lab",
  }, "EVERYTHING LIT IS SOMETHING THE SYNAPSE CAN MEASURE ITSELF"));
  svg.appendChild(heb);

  /* ============ BACKWARD CHANNEL (bp + fa) ============ */
  const backPath = `M ${XO} 214 L ${XO} 258 L ${XH} 258 L ${XH} ${HID[POST] + R + 2}`;

  const buildBack = (cls, boxCls, boxText, boxLab, extra) => {
    const g = el("g", { class: cls });

    /* the label the network is told */
    g.appendChild(el("rect", {
      x: XL - 20, y: 126, width: 42, height: 24, rx: 2,
      fill: "none", class: "lbl-box", "stroke-width": 1,
    }));
    g.appendChild(el("text", {
      x: XL + 1, y: 142, "text-anchor": "middle", class: "mach-lab",
    }, "LABEL"));
    g.appendChild(el("path", {
      d: `M ${XL - 22} 138 L ${XO + R + 3} 138`,
      class: "err-wire" + (cls === "fa-only" ? " rand" : ""),
    }));
    g.appendChild(el("text", {
      x: XL + 1, y: 166, "text-anchor": "middle", class: "lab",
    }, "δ = ŷ − y"));

    /* the wire back */
    g.appendChild(el("path", {
      d: backPath, class: "err-wire" + (cls === "fa-only" ? " rand" : ""),
      "marker-end": "url(#arw" + (cls === "fa-only" ? "-fa" : "") + ")",
    }));

    /* THE BOX. This is the only thing that differs between the two. */
    const bx = 296;
    g.appendChild(el("rect", {
      x: bx, y: 246, width: 44, height: 24, rx: 2,
      class: boxCls, "stroke-width": 1.5,
    }));
    g.appendChild(el("text", {
      x: bx + 22, y: 262, "text-anchor": "middle",
      class: cls === "fa-only" ? "mid-lab" : "flag-lab",
      "font-size": "10",
    }, boxText));
    g.appendChild(el("text", {
      x: bx + 22, y: 288, "text-anchor": "middle",
      class: cls === "fa-only" ? "mid-lab" : "flag-lab",
      "font-size": "8", "letter-spacing": "0.1em",
    }, boxLab));
    if (extra) g.appendChild(extra(bx));

    return g;
  };

  /* arrowheads */
  const defs = el("defs");
  [["arw", "mk-mach"], ["arw-fa", "mk-mid"]].forEach(([id, c]) => {
    const m = el("marker", {
      id, viewBox: "0 0 8 8", refX: 6, refY: 4,
      markerWidth: 5, markerHeight: 5, orient: "auto-start-reverse",
    });
    m.appendChild(el("path", { d: "M 0 1 L 7 4 L 0 7 z", class: c }));
    defs.appendChild(m);
  });
  svg.appendChild(defs);

  /* --- BACKPROP: the transport wire --- */
  const bp = buildBack("bp-only", "box-bp", "W2ᵀ", "COPIED", (bx) => {
    const g = el("g");
    /* the copy arc: values siphoned out of the forward W2 edges */
    g.appendChild(el("path", {
      d: `M ${(XH + XO) / 2} 150 C ${(XH + XO) / 2} 200, ${bx + 22} 200, ${bx + 22} 244`,
      class: "transport pulse",
    }));
    g.appendChild(el("text", {
      x: 300, y: 344, "text-anchor": "middle", class: "flag-lab",
    }, "⚠ NO WIRE IN THE BRAIN CARRIES THIS COPY"));
    return g;
  });
  svg.appendChild(bp);

  /* --- FEEDBACK ALIGNMENT: no copy at all --- */
  const fa = buildBack("fa-only", "box-fa", "B", "RANDOM · FIXED", () => {
    const g = el("g");
    g.appendChild(el("text", {
      x: 300, y: 344, "text-anchor": "middle", class: "mid-lab",
    }, "✓ NOTHING IS COPIED — AND IT STILL LEARNS"));
    return g;
  });
  svg.appendChild(fa);

  mount.appendChild(svg);
  return svg;
}

function initScope() {
  const stage   = document.getElementById("scope-stage");
  const seg     = document.getElementById("scope-seg");
  const ledger  = document.getElementById("scope-ledger");
  const verdict = document.getElementById("scope-verdict");
  const mathBox = document.getElementById("scope-math");
  if (!stage) return;

  const svg = buildScope(stage);

  function setMode(mode) {
    svg.setAttribute("data-mode", mode);

    /* what is lit, and what is blacked out */
    const w2   = svg.querySelectorAll(".w2");
    const outs = svg.querySelectorAll(".out-node");
    const lit  = mode !== "hebbian";
    w2.forEach((n)   => n.classList.toggle("dim", !lit));
    outs.forEach((n) => n.classList.toggle("dim", !lit));

    const L = LEDGER[mode];
    ledger.innerHTML = L.rows.map((r) => `
      <div class="row" data-ok="${r.ok}">
        <span class="k">${r.k}</span>
        <span class="v">${r.v}</span>
      </div>`).join("");

    verdict.className = "scope-verdict" + (L.verdict === "ok" ? " ok" : "");
    verdict.innerHTML = `<b>${L.headline}</b>${L.detail}`;

    const R = RULES[mode];
    mathBox.innerHTML =
      `<span class="rl">${R.line}</span>` +
      `<span class="cap">${R.needs}</span>`;

    seg.querySelectorAll("button").forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.mode === mode))
    );
  }

  seg.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (b) setMode(b.dataset.mode);
  });

  setMode("hebbian");
}
