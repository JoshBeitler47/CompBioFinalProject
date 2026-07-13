/* ============================================================
   main.js — wiring only. All content comes from data.js.
   ============================================================ */

/* ---------- build the timeline + refs from data ---------- */
function buildTimeline() {
  const mount = document.getElementById("timeline");
  if (!mount) return;
  mount.innerHTML = TIMELINE.map((t) => `
    <div class="tl-item${t.have ? " have" : ""}">
      <div class="yr">${t.year}</div>
      <div>
        <div class="who">${t.who}${t.have ? '<span class="tag">in our repo</span>' : ""}</div>
        <div class="what">${t.what}</div>
      </div>
    </div>`).join("");
}

function buildRefs() {
  const mount = document.getElementById("refs");
  if (!mount) return;
  mount.innerHTML = REFS.map((r) => `
    <div class="r"><div class="y">${r.y}</div><div class="t">${r.t}</div></div>
  `).join("");
}

/* ---------- the hero strip: same numbers, up top ---------- */
function buildStrip() {
  const mount = document.getElementById("ladder-strip");
  if (!mount) return;
  mount.innerHTML = LADDER.mnist.rows.map((r) => `
    <div class="cell" style="--c:${r.color}">
      <span class="nm">${r.name}</span>
      <span class="vl">${r.pending ? "—" : r.mean.toFixed(1) + "%"}</span>
      <span class="sb">${r.pending ? "not yet run" : "±" + r.std.toFixed(2)}</span>
    </div>`).join("");
}

/* ---------- the ladder + its ledger ---------- */
function initLadder() {
  const mount = document.getElementById("ladder-chart");
  const seg = document.getElementById("ladder-seg");
  const cap = document.getElementById("ladder-cap");
  const led = document.getElementById("ladder-ledger");
  const ledName = document.getElementById("ladder-ledger-name");
  const ledRule = document.getElementById("ladder-ledger-rule");
  if (!mount) return;

  let which = "mnist";

  function showLedger(key) {
    const L = LEDGER[key];
    const row = LADDER[which].rows.find((r) => r.key === key);
    ledName.textContent = L.name;
    ledName.style.color = L.color;
    ledRule.textContent = row ? row.rule : "";
    led.innerHTML = L.rows.map((r) => `
      <div class="row" data-ok="${r.ok}">
        <span class="k">${r.k}</span><span class="v">${r.v}</span>
      </div>`).join("");
  }

  function draw() {
    drawLadder(mount, which, showLedger);
    cap.textContent = LADDER[which].caption;
    showLedger("hebbian");
  }

  seg.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    which = b.dataset.set;
    seg.querySelectorAll("button").forEach((x) =>
      x.setAttribute("aria-pressed", String(x.dataset.set === which))
    );
    draw();
  });

  return draw;
}

/* ---------- reveal on scroll; charts draw when they're seen
   so their grow-in animation is actually witnessed ---------- */
function initReveal(deferred) {
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      e.target.classList.add("in");
      const job = e.target.dataset.draw;
      if (job && deferred[job] && !e.target.dataset.drawn) {
        e.target.dataset.drawn = "1";
        deferred[job]();
      }
      io.unobserve(e.target);
    });
  }, { threshold: 0.15, rootMargin: "0px 0px -40px 0px" });

  document.querySelectorAll(".rise").forEach((n) => io.observe(n));
}

/* ---------- nav ---------- */
function initNav() {
  const links = [...document.querySelectorAll(".nav-links a")];
  const secs = links
    .map((a) => document.querySelector(a.getAttribute("href")))
    .filter(Boolean);

  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      links.forEach((a) =>
        a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id)
      );
    });
  }, { rootMargin: "-45% 0px -50% 0px" });

  secs.forEach((s) => io.observe(s));
}

/* ---------- go ---------- */
document.addEventListener("DOMContentLoaded", () => {
  buildTimeline();
  buildRefs();
  buildStrip();
  initScope();

  const drawLad = initLadder();

  initReveal({
    ladder:     () => drawLad && drawLad(),
    forgetting: () => drawForgetting(document.getElementById("forget-chart")),
    tuning:     () => drawTuning(document.getElementById("tuning-chart")),
  });

  initNav();
  initDigit();   // async: loads weights, then wakes the demo + filter explorer
});
