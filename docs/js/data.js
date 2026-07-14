/* ============================================================
   data.js — THE SINGLE SOURCE OF TRUTH FOR EVERY NUMBER ON
   THIS PAGE.

   Every value below is transcribed from
   context/session_summary_fixes_and_results.md, section 5.
   Nothing here is estimated, smoothed, or invented. If you
   re-run an experiment and get a new number, change it HERE
   and the whole site updates.
   ============================================================ */

const PALETTE = {
  backprop: "#5FD3E4",
  fa:       "#6FCBA8",
  hebbian:  "#E8B33C",
  random:   "#5E7480",
  flag:     "#D96A8B",
  ink:      "#E9F0F2",
  dim:      "#93A6AD",
  mute:     "#61757D",
};

/* ---------- the ladder: hidden-layer learning rule vs. accuracy ---------- */
/* MNIST + Fashion-MNIST, from the 5-seed sweeps (multi_seed_variability.py,
   fashion_mnist_variability.py). All four models share the same
   784 -> 400 -> 10 architecture and the same supervised linear readout;
   the ONLY thing that differs is how the hidden layer's weights are learned. */

const LADDER = {
  mnist: {
    label: "MNIST",
    caption: "handwritten digits · 5 seeds · 60k train / 10k test",
    rows: [
      { key: "random",   name: "Random",             mean: 85.66, std: 0.32, color: PALETTE.random,
        rule: "hidden layer frozen at its random init — never learns at all" },
      { key: "hebbian",  name: "Hebbian",            mean: 89.03, std: 0.50, color: PALETTE.hebbian,
        rule: "local, label-free: anti-Hebbian lateral inhibition + homeostatic thresholds" },
      { key: "backprop", name: "Backprop",           mean: 97.75, std: 0.05, color: PALETTE.backprop,
        rule: "textbook gradient descent — the error is routed back through W2ᵀ" },
    ],
  },
  fashion: {
    label: "Fashion-MNIST",
    caption: "clothing photos · 5 seeds · 60k train / 10k test",
    rows: [
      { key: "random",   name: "Random",             mean: 74.40, std: 0.37, color: PALETTE.random,
        rule: "hidden layer frozen at its random init — never learns at all" },
      { key: "hebbian",  name: "Hebbian",            mean: 73.80, std: 0.93, color: PALETTE.hebbian,
        rule: "local, label-free: anti-Hebbian lateral inhibition + homeostatic thresholds" },
      { key: "backprop", name: "Backprop",           mean: 88.42, std: 0.15, color: PALETTE.backprop,
        rule: "textbook gradient descent — the error is routed back through W2ᵀ" },
    ],
  },
};

/* ---------- the biological-plausibility ledger ----------
   Applies to how the HIDDEN LAYER's weights are learned.
   The linear readout on top is supervised in all four models —
   that's the bridge, and we say so out loud on the page. */

const LEDGER = {
  hebbian: {
    name: "Hebbian",
    color: PALETTE.hebbian,
    verdict: "ok",
    headline: "Nothing here is physically impossible.",
    detail: "Every term in the update is something the synapse itself can measure. This is the only model on the bench a real synapse could actually run.",
    rows: [
      { k: "Local update only",     v: "YES", ok: "yes" },
      { k: "Needs no labels",       v: "YES", ok: "yes" },
      { k: "No weight transport",   v: "YES", ok: "yes" },
      { k: "No separate backward pass", v: "YES", ok: "yes" },
    ],
  },
  backprop: {
    name: "Backprop",
    color: PALETTE.backprop,
    verdict: "bad",
    headline: "A synapse would have to read a weight it has never touched.",
    detail: "To know how much it contributed to the error, this synapse needs the values of the output-layer weights. There is no wire in the brain that delivers them. This is the weight transport problem.",
    rows: [
      { k: "Local update only",     v: "NO", ok: "no" },
      { k: "Needs no labels",       v: "NO", ok: "no" },
      { k: "No weight transport",   v: "NO", ok: "no" },
      { k: "No separate backward pass", v: "NO", ok: "no" },
    ],
  },
  random: {
    name: "Random control",
    color: PALETTE.random,
    verdict: "na",
    headline: "It never learns, so it never breaks a rule.",
    detail: "The hidden layer keeps its random initialisation forever. It exists to answer one question: did the Hebbian layer actually learn anything, or would any 400 random directions have done just as well?",
    rows: [
      { k: "Local update only",     v: "N/A", ok: "na" },
      { k: "Needs no labels",       v: "YES", ok: "yes" },
      { k: "No weight transport",   v: "YES", ok: "yes" },
      { k: "No separate backward pass", v: "YES", ok: "yes" },
    ],
  },
};

/* ---------- catastrophic forgetting: Split-MNIST, 2 seeds ----------
   Train on digits 0-4, then train on 5-9, then re-test on 0-4.
   "fresh" = retrain ONLY the linear readout on the (drifted) features
   and re-test — i.e. did the representation survive, or just the readout? */

const FORGETTING = {
  caption: "Split-MNIST · train 0–4, then 5–9, then re-test 0–4 · 2 seeds",
  rows: [
    { name: "Backprop",          color: PALETTE.backprop, before: 98.93, after: 0.63, fresh: 98.37, lost: 98.30 },
    { name: "Backprop (sparse)", color: PALETTE.backprop, before: 98.92, after: 4.37, fresh: 98.43, lost: 94.55, sparse: true },
    { name: "Hebbian",           color: PALETTE.hebbian,  before: 97.57, after: 1.90, fresh: 96.57, lost: 95.67 },
    { name: "Hebbian (sparse)",  color: PALETTE.hebbian,  before: 97.54, after: 2.26, fresh: 96.66, lost: 95.28, sparse: true },
    { name: "Random (frozen)",   color: PALETTE.random,   before: 92.74, after: 0.01, fresh: 92.70, lost: 92.73 },
  ],
};

/* ---------- the second cost: the sparsity knob is task-dependent ----------
   Same "tuned" hyperparameters (target_rate 0.08 -> 0.20, threshold_lr 0.3 -> 0.15)
   help on MNIST and hurt on Fashion-MNIST. Backprop needs no such knob. */

const TUNING = [
  { task: "MNIST",         base: 89.03, baseStd: 0.50, tuned: 89.21, tunedStd: 0.40 },
  { task: "Fashion-MNIST", base: 73.80, baseStd: 0.93, tuned: 72.26, tunedStd: 0.32 },
];

/* ---------- representation richness ----------
   Effective dimensionality of the 400-unit hidden layer, MNIST
   (compare_and_visualize_v2.py, 3 seeds). */

const EFFDIM = {
  units: 400,
  hebbian: 108,
  random: 347,
};

/* ---------- the historical spine ----------
   `have: true` = the paper is sitting in this repo. */

const TIMELINE = [
  { year: "1943", who: "McCulloch & Pitts", have: true,
    what: "A neuron is a logic gate: it fires all-or-none once its inputs cross a threshold. The artificial neuron is born as a claim about <em>biology</em>, not a claim about engineering." },
  { year: "1949", who: "Donald Hebb",
    what: "Proposes that when one cell repeatedly helps fire another, the connection between them strengthens. He has no evidence. He is right anyway." },
  { year: "1958", who: "Frank Rosenblatt", have: true,
    what: "The perceptron: a machine that <em>learns</em> its own weights from examples, built in hardware. Rosenblatt cites Hebb directly." },
  { year: "1962", who: "Hubel & Wiesel", have: true,
    what: "Cells in the cat's visual cortex fire for oriented edges. Biology hands machine learning the idea of a <em>learned feature detector</em> — the thing our hidden layer is." },
  { year: "1973", who: "Bliss & Lømo",
    what: "Long-term potentiation found in the hippocampus. Hebb's 24-year-old guess turns out to be a real, measurable, molecular mechanism." },
  { year: "1980", who: "Kunihiko Fukushima", have: true,
    what: "The Neocognitron learns its feature layers with <em>unsupervised competition</em> and puts a supervised readout on top. That is our architecture — 45 years early." },
  { year: "1982", who: "Erkki Oja",
    what: "Adds a brake to Hebb's rule using only what a synapse can see. A single neuron running it performs online PCA. Locality and real computation are compatible." },
  { year: "1986", who: "Rumelhart, Hinton & Williams",
    what: "Backpropagation is popularised. It works spectacularly, and it asks biology for no permission at all. Everything modern descends from this." },
  { year: "2016", who: "Lillicrap et al.",
    what: "Feedback alignment: replace the backward weights with fixed random ones. It <em>still learns</em>. Weight transport was never the load-bearing part." },
  { year: "2019", who: "Krotov & Hopfield",
    what: "A Hebbian rule with strong competition gets far closer to backprop on MNIST than ours does. The gap we measure is not a law of nature." },
];

const REFS = [
  { y: "1943", t: "<b>McCulloch, W. S., & Pitts, W.</b> A logical calculus of the ideas immanent in nervous activity. <i>Bulletin of Mathematical Biophysics</i>, 5, 115–133." },
  { y: "1949", t: "<b>Hebb, D. O.</b> <i>The Organization of Behavior.</i> Wiley, New York." },
  { y: "1958", t: "<b>Rosenblatt, F.</b> The perceptron: a probabilistic model for information storage and organization in the brain. <i>Psychological Review</i>, 65(6), 386–408." },
  { y: "1962", t: "<b>Hubel, D. H., & Wiesel, T. N.</b> Receptive fields, binocular interaction and functional architecture in the cat's visual cortex. <i>Journal of Physiology</i>, 160, 106–154." },
  { y: "1973", t: "<b>Bliss, T. V. P., & Lømo, T.</b> Long-lasting potentiation of synaptic transmission in the dentate area of the anaesthetized rabbit. <i>Journal of Physiology</i>, 232, 331–356." },
  { y: "1980", t: "<b>Fukushima, K.</b> Neocognitron: a self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position. <i>Biological Cybernetics</i>, 36, 193–202." },
  { y: "1982", t: "<b>Oja, E.</b> A simplified neuron model as a principal component analyzer. <i>Journal of Mathematical Biology</i>, 15, 267–273." },
  { y: "1986", t: "<b>Rumelhart, D. E., Hinton, G. E., & Williams, R. J.</b> Learning representations by back-propagating errors. <i>Nature</i>, 323, 533–536." },
  { y: "1989", t: "<b>McCloskey, M., & Cohen, N. J.</b> Catastrophic interference in connectionist networks: the sequential learning problem. <i>Psychology of Learning and Motivation</i>, 24, 109–165." },
  { y: "1990", t: "<b>Földiák, P.</b> Forming sparse representations by local anti-Hebbian learning. <i>Biological Cybernetics</i>, 64, 165–170." },
  { y: "1995", t: "<b>McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C.</b> Why there are complementary learning systems in the hippocampus and neocortex. <i>Psychological Review</i>, 102(3), 419–457." },
  { y: "1998", t: "<b>Turrigiano, G. G., et al.</b> Activity-dependent scaling of quantal amplitude in neocortical neurons. <i>Nature</i>, 391, 892–896." },
  { y: "1999", t: "<b>Rao, R. P. N., & Ballard, D. H.</b> Predictive coding in the visual cortex. <i>Nature Neuroscience</i>, 2, 79–87." },
  { y: "2016", t: "<b>Lillicrap, T. P., Cownden, D., Tweed, D. B., & Akerman, C. J.</b> Random synaptic feedback weights support error backpropagation for deep learning. <i>Nature Communications</i>, 7, 13276." },
  { y: "2017", t: "<b>Whittington, J. C. R., & Bogacz, R.</b> An approximation of the error backpropagation algorithm in a predictive coding network with local Hebbian synaptic plasticity. <i>Neural Computation</i>, 29(5), 1229–1262." },
  { y: "2019", t: "<b>Krotov, D., & Hopfield, J. J.</b> Unsupervised learning by competing hidden units. <i>PNAS</i>, 116(16), 7723–7731." },
];
