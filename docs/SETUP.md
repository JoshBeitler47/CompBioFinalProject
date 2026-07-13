# The website

A single-page site covering both halves of the assignment: the research review
(history, biology, the math, the plausibility argument) and the results
(every chart drawn live from your own numbers), plus two interactives that run
your real trained models in the browser.

## Deploy

**GitHub Pages, zero config.** Settings → Pages → *Deploy from a branch* →
`main` / `/docs`. That's it. Live at
`https://<user>.github.io/CompBioFinalProject/`.

## Two things to do

### 1. One missing number

`models/feedback_alignment.py` prints its accuracy but never saved a figure, so
the number never made it into the notes — and feedback alignment turns out to be
the most important bar on the main chart. Run it, then open
`docs/js/data.js` and fill in:

```js
const FEEDBACK_ALIGNMENT = {
  mnist:   { mean: null, std: null, pending: true },   // ← put it here
  fashion: { mean: null, std: null, pending: true },
};
```

Set `mean`, set `std` (or `0` if you only ran one seed), and delete
`pending: true`. Every chart, table, and sentence that mentions feedback
alignment fills itself in. Until then the site renders it honestly as a hatched
"not yet run" bar rather than guessing — which is defensible if you run out of
time, but the argument is much stronger with the number in.

Fashion-MNIST feedback alignment is optional. If you don't run it, leave that
line as-is; only the Fashion tab will show the hatched bar.

### 2. The browser demo needs weights

```sh
python models/export_web_assets.py          # a few minutes on CPU
python -m http.server --directory docs 8000 # then open localhost:8000
```

That writes `docs/assets/models.json` (~1.4 MB) — int8-quantised weights for the
backprop MLP and the Hebbian layer. **Commit it**; Pages has to serve it.

Until it exists, the draw-a-digit panel and the filter explorer show a clear
"weights not loaded" state instead of breaking. Everything else on the page works
without it.

> **Opening `index.html` by double-clicking will not work.** Browsers block
> `fetch()` on `file://` URLs, so the weights can't load. Use the local server
> above, or just push to Pages.

The export script uses the repo's own `models/common.py` functions directly, so
what it exports is the exact same Hebbian rule and weights the rest of the repo
produces — not a reimplementation.

## How it's wired

Every number on the page lives in **`docs/js/data.js`** and nowhere else —
transcribed from `context/session_summary_fixes_and_results.md`. Change a number
there and the whole site updates. Nothing is hardcoded into a chart, and no
figure is a static image: the charts are SVG drawn at page load, so they stay in
sync with the data by construction.

| File | Does |
|---|---|
| `js/data.js` | all results, the timeline, the references. **The only file with numbers in it.** |
| `js/charts.js` | the ablation ladder, the forgetting chart, the tuning slope |
| `js/synapse.js` | the synapse's-eye view — the interactive network diagram |
| `js/net.js` | loads the weights, runs the forward passes |
| `js/digit.js` | the drawing pad (with MNIST's own preprocessing recipe) |
| `js/filters.js` | reshapes each hidden unit's 784 weights back into a 28×28 image |
| `js/main.js` | wiring |

## The ten sections = the ten beats of the talk

1. **Hero** — your brain cannot run backprop; here are the four numbers
2. **Draw one** — live demo, both models arguing
3. **The question** — the two rules, and *the bridge* (all four share a supervised readout — stated out loud, because a sharp grader will ask)
4. **The synapse's-eye view** — the signature. Black out everything a synapse can't physically read, then ask each rule to do its job
5. **History** — 1943 – 2019, with the four papers from your repo on it
6. **The ladder** — the main result, MNIST/Fashion toggle, hover a bar to see what it costs biologically
7. **What it learned** — the filter explorer; the plausible model is the readable one
8. **Where it breaks** — Fashion-MNIST, and the tuning knob that has no right answer
9. **Forgetting** — the negative result, and the control that caught it
10. **So what** — predictive coding, 20 watts, and what you'd fix

## A note on what I did and didn't claim

The site says lateral inhibition **didn't** turn out to be the mechanism, that
your forgetting hypothesis was **wrong**, and that the Krotov & Hopfield rule
beats yours. Those are all in your notes, and leaving them in is what makes the
rest credible. The "we were wrong" section is genuinely one of the strongest
things on the page — don't let anyone talk you into cutting it.

Nothing is charted that isn't in your notes. Sample-efficiency and training-curve
figures exist in `output/` but their raw numbers aren't in the write-up, so they
were left out rather than eyeballing values off a PNG.
