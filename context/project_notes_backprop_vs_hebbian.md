# Project Notes: Backpropagation vs. Biologically-Inspired Learning

*Working notes for the CMU pre-college computational biology final project.*
*Focus question: How does backpropagation differ from learning in the human brain — specifically from Hebbian-style learning — and how do the two compare when trained on MNIST?*

---

## 1. Evaluating the Groupmate's Three Angles

He's onto something. All three ideas are real, well-grounded in the neuroscience/ML literature, and any of them pushes the project past merely re-confirming "backprop isn't biologically plausible." His instinct about *how to use* each one is also mostly right. But each has a catch, and the biggest risk is trying to do all three as deep dives.

### Angle 1 — Catastrophic forgetting (the testable one)

**What's real:** Well-documented (McCloskey & Cohen 1989; French 1999). The "two memory systems + sleep replay" story is Complementary Learning Systems theory (McClelland, McNaughton & O'Reilly 1995) — hippocampus learns fast, neocortex consolidates slowly, replay interleaves old memories so new learning doesn't overwrite old.

**The experiment:** Standard benchmark called **Split MNIST**. Train on digits 0–4, then train on 5–9, then measure how far accuracy on 0–4 collapsed. Very feasible.

**The catch he glossed over:** It is *not* obvious that a Hebbian model forgets less. Plain Hebbian weights get overwritten too — unstably, even (which is why Oja's rule exists). What actually reduces forgetting in the brain is mostly:
- **sparse, non-overlapping representations** — if digits 0–4 and 5–9 activate mostly *different* neurons, they can't clobber each other
- **replay**

That means sparsity, not Hebbian-ness *per se*, may be doing the work. If the Hebbian model forgets less, we won't know whether that's the **learning rule** or the **sparsity**.

**The rigorous version:** control for it — compare sparse vs. dense backprop nets, and sparse vs. dense Hebbian. Disentangling those two is a *more* sophisticated result than "Hebbian forgets less," and it's still tractable.

### Angle 2 — Predictive coding (keep as framing, don't build it)

**What's real:** Rao & Ballard (1999) on the neuroscience side; Whittington & Bogacz (2017) showed predictive coding networks can approximate backprop.

**Why it's a good move:** His line — *"Hebbian hardware doing backprop's job"* — is a fair summary of a live NeuroAI thesis. Conceptually it **dissolves the false dichotomy** the whole project is built on. Instead of "two rivals, backprop wins," it becomes "here's how the field is trying to get backprop's power out of local, brain-like machinery." Much more grown-up framing for the discussion section.

**The catch:** Full predictive coding is hard to implement and almost certainly out of scope.

**Better code option if we want a third model — feedback alignment** (Lillicrap et al. 2016). It's a tiny change from standard backprop: replace the transposed weight matrix in the backward pass with a *fixed random matrix*. It directly attacks the **weight transport problem** already central to our biological-plausibility argument.

> **Recommendation:** predictive coding = the *story* in the writeup. Feedback alignment = the *code*, if we want a third data point.

### Angle 3 — The 20-watt angle (great narrative, don't overclaim)

**What's real:** The ~20W figure is right. The link from energy → **sparsity** is solid (spikes cost ATP; firing less saves real metabolic money — see the energy-efficient-coding literature). As connective tissue for a 10-minute talk it's excellent and memorable, especially contrasted with the megawatts needed to train a large model.

**The hedge:** "The brain is local *because* of 20 watts" is a bit of a just-so story. **Locality is driven more by biophysics than by energy** — a synapse can only "see" its own pre- and post-synaptic neuron; there's no wire delivering a global error signal to every synapse regardless of budget. You could imagine an energy-rich brain that's still local for pure wiring reasons.

So: **sparsity has the clean energy link; locality is a separate constraint** that the energy frame lumps in. Present it as "a useful lens that ties these together," not "the single cause," and it holds up fine.

### Scope recommendation

Three new angles on top of the existing backprop-vs-Hebbian-on-MNIST plan is a lot for one summer project and a 10-minute talk. Failure mode = sprawling talk with no through-line. Sharpened version of his own prioritization:

- **Spine:** the core MNIST comparison (already planned).
- **One new experiment:** catastrophic forgetting / Split MNIST — it extends what we're already building and it's the empirically novel bit.
- **Framing in the discussion:** predictive coding + the energy argument, *not* as separate investigations.

**Resulting single narrative:**
> Backprop wins on accuracy but forgets catastrophically and costs enormous energy. The brain trades some accuracy for lifelong, local, 20-watt learning. Predictive coding and feedback alignment are how the field is trying to close that gap.

### Bonus historical tie-in (already in our project folder)

**Fukushima's Neocognitron (1980)** trained its feature layers with unsupervised, competitive self-organization — essentially Hebbian-flavored learning — *before* backprop-trained CNNs took over. This:
- anchors the Hebbian thread historically, and
- shows the "local feature learning + supervised readout" bridge isn't something we invented for the project — it's how one of the ancestor architectures actually worked.

---

## 2. Is the Biology of the Brain Really Hebbian?

Yes — more than the cartoonish slogan suggests. But it's a **first-order approximation**, and the ways it's *wrong* are exactly the ways that matter for this project.

### Where Hebb was genuinely vindicated

Hebb wrote his postulate in **1949 with no direct evidence**. Then:

- **LTP discovered** in the hippocampus (Bliss & Lømo, 1973): stimulate a pathway at high frequency and the synapse gets durably stronger.
- **The molecular mechanism is almost embarrassingly Hebbian.** The **NMDA receptor is a coincidence detector**: it only opens if glutamate is present (presynaptic cell fired) **and** the postsynaptic cell is depolarized enough to expel a magnesium ion blocking the channel (postsynaptic cell fired). It is a literal **molecular AND gate** for "both cells active together."

Hebb predicted a mechanism decades before anyone found it, and it's real and load-bearing.

### Where the cartoon breaks down

| Issue | Reality |
|---|---|
| **Weakening exists** | Hebb described only strengthening. Real synapses also undergo **LTD**. Pure "fire together, wire together" with no depression runs away to saturation — precisely why Oja's rule exists computationally. |
| **Timing/order matter** | **STDP**: if the pre-spike arrives ~10–20 ms *before* the post-spike, strengthen; flip the order and it weakens. The rule is **causal**, not merely correlational — closer to what Hebb actually wrote ("takes part in firing it") than the slogan he's remembered for. |
| **Homeostasis** | Neurons monitor their own average activity and scale all synapses up/down to stay in range (**synaptic scaling**, Turrigiano). The biological version of the normalization you'd have to bolt onto a Hebbian model. |
| **Neuromodulation** | Whether plasticity happens *at all* is often gated by dopamine, acetylcholine, noradrenaline. |

### The three-factor rule — and why it sharpens our thesis

Modern formulation:

> **Δw ∝ pre × post × neuromodulatory signal**

The synapse holds a decaying **eligibility trace** of recent coincident activity, and a later reward/surprise signal decides whether that trace gets consolidated.

**This complicates the standard objection our project is built on.** The usual claim is *"backprop needs a global error signal, and the brain doesn't have one."* **That's too strong** — dopamine broadcasts something a lot like a global error signal (Schultz's reward prediction error).

What the brain plausibly *doesn't* have:
1. a **per-synapse, precisely credited** error signal, and
2. the **weight transport** needed to compute it (each synapse would need to know the weights of downstream synapses it never touches).

**Sharpening the objection from "no global signal" → "no per-synapse credit assignment, no weight transport"** makes the argument both more accurate and more interesting — and it's the exact gap **feedback alignment** was invented to probe.

### The honest framing for the presentation

Neither model on our bench is "the brain."
- Backprop is biologically implausible in **specific, nameable** ways.
- But our Hebbian model is a **drastic simplification of biology too** — real synapses are Hebbian-ish machinery wrapped in a lot of stabilizing and gating apparatus that a two-term update rule throws away.

Saying that out loud signals we've read the literature rather than the slogan.

---

## 3. What Is a Synapse? (baseline refresher)

A **synapse** is the junction where one neuron passes a signal to another. It's the connection point.

**The basic picture:** a neuron takes in signals through branches called **dendrites**. If it gets excited enough, it fires an electrical spike down a long cable called an **axon**. The axon reaches the dendrite of the *next* neuron — that meeting point is the synapse.

They don't actually touch. There's a tiny gap. When the spike arrives, the sending neuron dumps chemicals (**neurotransmitters**) across the gap; the receiving neuron has receptors that catch them and turn that back into an electrical nudge.

- **presynaptic** = the sending neuron (before the gap)
- **postsynaptic** = the receiving neuron (after the gap)

That's where "pre" and "post" in all the learning rules come from.

### Why it's the star of the project

Synapses aren't all equally strong. Some deliver a big nudge, some a weak one. That strength is the **synaptic weight**, and it changes with experience. **That change is learning.** Everything you know is, physically, a pattern of synaptic strengths.

**In an artificial neural network, all the biology drops away and a synapse is just a number in a matrix.** That number is the weight. "Training a network" means adjusting those numbers. Backprop and Hebbian learning are two different answers to the same question — *given what just happened, how should each of these numbers change?*

- **Hebbian:** if the sending and receiving units were both active, make the number bigger.
- **Backprop:** figure out how much this particular number contributed to the network's mistake, and nudge it to reduce that mistake.

### Where the biological-plausibility fight lives

A real synapse is a physical object stuck between two cells. The only things it can "see" are:
1. its own sending neuron,
2. its own receiving neuron,
3. its own current strength.

Hebbian rules use **only that** — which is why they're called **local**. Backprop's update needs information from far away in the network (weights of other synapses it has no contact with), which a physical synapse has no way of knowing.

**That's the whole objection in one sentence.**

---

## 4. Oja's Rule

The fix for the single biggest practical problem with plain Hebbian learning — and probably the rule we'll actually implement.

### 4a. The plain-English version

**The problem.** Plain Hebbian says: when two neurons fire together, strengthen the connection. Sounds fine — until you notice it's **a loop with no brake**. A stronger connection makes the second neuron fire harder → the connection strengthens more → it fires even harder. The weights blow up to infinity. The rule can only ever say "more," never "less."

**The obvious fix.** After each update, shrink all weights back down so they add to a fixed total. Like a budget: if one connection grows, others give something back.

**Why that fix fails biologically.** To know how much to shrink by, a synapse would have to know the total across *all* the neuron's other synapses. A synapse is a tiny junction between two cells — it can't poll the neuron's thousands of other connections. **Not something a real brain could implement.**

**Oja's trick.** He found a way to get almost exactly that budget effect using only information the synapse already has:

> **Strengthen the connection when both cells fire together — but also subtract a little, in proportion to how hard the output neuron just fired.**

That subtraction is the brake. Fire hard, and all your connections shrink slightly. It automatically keeps the total in check. And each synapse can compute it **alone**: it needs its input's activity, its output's activity, and its own current strength. Nothing else. **Normalization, achieved locally.**

**What you get.** The growth term pushes the neuron toward whatever pattern is most common in its input; the brake stops it running away. So a neuron running Oja's rule settles onto **the single strongest pattern of variation in what it's been shown.** Show it lots of faces, and it tunes to the biggest way faces differ from each other. That's **principal component analysis** — a real statistical technique — falling out of a dumb local rule that never sees more than one example at a time.

### 4b. IMPORTANT CLARIFICATION — the subtraction is *not* a punishment for firing together

This is easy to misread. The key point:

> **The two terms depend on different things.**

- The **growth** term needs **both** input and output to be active. It only fires for synapses that actually participated.
- The **shrink** term depends **only on the output neuron** (and the synapse's own current strength). It doesn't care what the input did.

So the shrink hits **every** synapse on that neuron — including all the ones whose input was silent.

Play it out for a neuron that just fired hard:

| Synapse | Boost? | Shrink? | Net |
|---|---|---|---|
| Input **was** active | ✅ big boost | small shrink | **grows** |
| Input was **silent** | ❌ no boost | small shrink | **shrinks** |

**Co-firing is still rewarded.** The reward is just now funded by **a tax that everyone pays**. Participants come out ahead; non-participants come out behind. Nobody is punished *for* firing together — they're punished for *not participating* while the neuron fired anyway.

**That's what makes it a competition.** Under plain Hebbian, every synapse can grow at once and they all blow up together. Under Oja, growth for one connection effectively comes at the expense of others, so the neuron is forced to **choose what it cares about** instead of saying yes to everything.

And the "grows" case has a natural stopping point: once a synapse is strong enough that the tax matches the boost, it stops growing. **It settles rather than running away.**

### 4c. The math

Linear neuron: output `y = wᵀx`.

**Plain Hebbian:**
```
Δw = η · y · x
```
Substitute `y` and average over data:
```
⟨Δw⟩ = η · ⟨x xᵀ⟩ · w = η · C · w
```
where `C` = input correlation matrix. **That's power iteration** — every update multiplies `w` by a positive-definite matrix, so weights grow without bound. Positive feedback with no brake.

**The non-local fix** (renormalize every step):
```
w ← (w + η·y·x) / ‖w + η·y·x‖
```
Works, but requires knowing `‖w‖` — i.e. every other synapse on the neuron.

**Oja's derivation** — Taylor-expand that normalization for small η, assuming `‖w‖ = 1`:
```
‖w + η y x‖ ≈ 1 + η·y·(wᵀx) = 1 + η·y²

w' ≈ (w + η y x)(1 − η y²) ≈ w + η y x − η y² w
```
Drop the O(η²) term:

> ### **Δw = η · y · (x − y·w)**

Per synapse: **Δwᵢ = η · y · (xᵢ − y · wᵢ)**

It needs: the presynaptic activity `xᵢ`, the postsynaptic activity `y`, and **its own** current weight `wᵢ`. **Nothing else.** Normalization smuggled in as a local decay term. **That locality is the whole reason it belongs in this project.**

### 4d. Two ways to read the new term

**(1) As forgetting.** The `−η y² wᵢ` term is a decay proportional to how strongly the neuron just fired. Fire hard → all synapses shrink a bit. A weight-dependent brake that pins `‖w‖` near 1. *Biologically this is in the neighborhood of heterosynaptic depression / synaptic scaling.*

**(2) As reconstruction error.** Rewrite as:
```
Δwᵢ ∝ y · (xᵢ − x̂ᵢ)     where   x̂ᵢ = y · wᵢ
```
`x̂ᵢ` is the neuron's attempt to **reconstruct its own input** from its output. So **Oja's rule is Hebbian learning on the residual** — learn from what you failed to predict.

**That's a baby version of predictive coding** — a direct bridge to the groupmate's angle #2. Worth a sentence in the writeup.

### 4e. What it converges to

Set expected update to zero:
```
C·w = (wᵀ C w)·w
```
So `w` is a **unit-norm eigenvector of C**. Stability analysis picks out exactly one: the eigenvector with the **largest eigenvalue**.

> **A single Oja neuron performs online PCA** — it finds the first principal component of its input stream, one sample at a time, using only local information, never storing a covariance matrix.

A dumb local synaptic rule computing a global statistical quantity. Lovely result, good thing to show the audience.

### 4f. Scaling to many neurons

One Oja neuron = one direction. Run N of them independently and **they all converge to the same PC** — useless. Options:

- **Sanger's rule (Generalized Hebbian Algorithm):** subtract earlier neurons' contributions so neuron *k* gets the *k*-th principal component. Slightly less local.
- **Oja's subspace rule:** neurons span the top-*k* PC subspace (but not individual PCs).
- **Anti-Hebbian lateral connections (Földiák):** hidden units inhibit each other with a rule that *weakens* on co-activity, forcing them to decorrelate and pick different features. **Most biologically appealing option**, and matches the "Hebbian/anti-Hebbian" bridge already in the project plan.

---

## 5. How Oja's Rule Relates to Real Brain Biology

### The part that's real biology

Real synapses *do* get weaker, not just stronger. Two mechanisms do roughly what Oja's shrink term does:

- **Heterosynaptic depression.** When a neuron fires hard, synapses that *didn't* participate get weakened. **That's almost exactly Oja's tax** — the shrink hits everyone, but only the participants earn it back. Real, observed, and it means synapses on the same neuron genuinely **compete for a limited budget**.

- **Synaptic scaling.** Neurons watch their own average firing rate over hours/days. Firing too much → turn all synapses down; too little → turn them all up. **A thermostat.** Same job as Oja's brake: keep the neuron in a sane operating range so it neither saturates nor goes silent.

So the *idea* Oja captured — **Hebbian growth needs a built-in brake, and that brake must be something a synapse can compute on its own** — is correct, and the brain has solved it. Hebb described only growth; Oja said "that can't be the whole story," and biology agrees.

### The part that isn't

The exact rule is **not** what a synapse does.

- Oja's specific formula (shrink ∝ output **squared** × own weight) is **derived from math, not measured from cells**. It answers *"what's the neatest local rule that yields PCA?"*, not *"what do neurons do?"*
- Real plasticity depends on **spike timing** (STDP) — did the input arrive just *before* the output, or after? **Oja's rule has no clock**; it only sees activity levels.
- Real plasticity is **gated by neuromodulators** like dopamine — the brain decides whether a moment was worth learning from at all. **Oja's rule always learns, indiscriminately.**
- Synaptic scaling is **slow** (hours); Oja's brake acts on **every single input**. Different timescales entirely.
- **Nothing in the brain is trying to compute principal components.** That's an outcome Oja *proved about his rule*, not a goal any neuron has.

### The line to use in the presentation

> **Oja's rule isn't a model of a synapse. It's a model of a *constraint* that synapses are under:**
> **grow on coincidence, but stay bounded, using only what's locally available.**
> The brain obeys that constraint. Oja found one clean, mathematical way to obey it.

This is worth saying out loud because it makes the same point as the project's whole thesis: **our Hebbian model is a simplification of biology, just as backprop is a departure from it. Neither thing on our bench is a brain.** What's biologically real is the **constraint — locality** — and that's the axis our comparison is actually measuring.

---

## 6. Practical Notes for the MNIST Code

1. **Center the data.** Subtract the mean image. Otherwise the first principal component is basically "the average digit" and you waste a unit on it. This trips people up constantly.

2. **Expected result.** "Oja features + supervised linear readout" ≈ "PCA + logistic regression" → roughly **low 90s** on MNIST, vs. **~98%** for a small backprop MLP. That's a perfectly good result: a real, characterizable gap, and exactly the comparison the focus question asks for.

3. **Be honest about *why* it underperforms.** **PCA maximizes variance, not class discriminability.** Nothing in Oja's rule has ever heard of the label "7." The directions of maximum variance in MNIST are largely **stroke thickness and slant** — which aren't what separates a 3 from an 8. Backprop's features are shaped by the **error signal**; Oja's are shaped only by the **input statistics**.

   > **That is the credit-assignment gap made concrete** — and a far sharper point than "backprop got a higher number."

   Framed simply: **features shaped by the *input* vs. features shaped by the *task*.** That gap is the concrete thing our comparison measures.

4. **Don't overstate the gap.** A much more competitive Hebbian baseline exists: **Krotov & Hopfield (2019), "Unsupervised learning by competing hidden units"** — a Hebbian-with-competition rule that gets much closer to backprop on MNIST. **Cite it even if we don't implement it.**

5. **Keep the backprop baseline a small MLP**, not a CNN, so the comparison stays apples-to-apples (the Hebbian model won't be convolutional).

---

## 7. Key References

**Foundational / historical**
- McCulloch & Pitts (1943) — logical calculus of neural activity *(in project folder)*
- Hebb (1949) — *The Organization of Behavior*; the Hebbian postulate
- Rosenblatt (1958) — the perceptron *(in project folder)*
- Oja (1982) — a simplified neuron model as a principal component analyzer
- Fukushima (1980) — Neocognitron; unsupervised competitive feature learning *(in project folder)*

**Biology of plasticity**
- Bliss & Lømo (1973) — discovery of LTP
- Turrigiano — synaptic scaling / homeostatic plasticity
- STDP literature — spike-timing-dependent plasticity
- Schultz — dopamine as reward prediction error

**Biological plausibility of backprop**
- Lillicrap et al. (2016) — feedback alignment; the weight transport problem
- Rao & Ballard (1999) — predictive coding in visual cortex
- Whittington & Bogacz (2017) — predictive coding approximates backprop
- Krotov & Hopfield (2019) — unsupervised learning by competing hidden units

**Catastrophic forgetting**
- McCloskey & Cohen (1989) — catastrophic interference
- French (1999) — catastrophic forgetting in connectionist networks
- McClelland, McNaughton & O'Reilly (1995) — Complementary Learning Systems

---

## 8. Open Next Steps

- [ ] Decide final scope (recommended: MNIST spine + Split-MNIST forgetting experiment + predictive-coding/energy framing in discussion)
- [ ] Map the project-folder papers (Fukushima, McCulloch–Pitts, Rosenblatt, the physiology paper) onto the research thread
- [ ] Implement Oja + linear readout on MNIST
- [ ] Implement backprop MLP baseline
- [ ] Optional third model: feedback alignment
- [ ] Split-MNIST forgetting test, with a sparsity control to disentangle rule vs. sparsity
- [ ] Website: accuracy plots, confusion matrices, misclassified examples
- [ ] Possible visualization: Oja's weight vector converging to the leading eigenvector
