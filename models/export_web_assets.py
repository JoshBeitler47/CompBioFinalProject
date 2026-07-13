"""
export_web_assets.py
====================
Writes `docs/assets/models.json` -- the weight file the website loads so the
draw-a-digit demo and the filter explorer can run the real networks in the
browser, with no server and no API call.

    Run from the REPOSITORY ROOT:

        python models/export_web_assets.py

    Then serve the site (browsers block fetch() on file:// URLs):

        python -m http.server --directory docs 8000
        # open http://localhost:8000

Takes a few minutes on CPU. Prints the test accuracy of everything it exports.

--------------------------------------------------------------------------
THIS USES THE REPO'S OWN TRAINING CODE
--------------------------------------------------------------------------
Every model here is trained with the exact same functions the rest of the
comparison suite uses (`common.load_dataset`, `common.hebbian_W1`,
`common.hebbian_features`, `common.train_backprop`, `common.train_readout`),
so what gets exported is the real Hebbian rule (Foldiak-style anti-Hebbian
lateral inhibition + Turrigiano-style homeostasis, mode="lateral", the
default) and the real backprop baseline -- not a reimplementation that could
silently drift from what `models/common.py` actually does. If `common.py`
changes, re-running this script picks the change up automatically.

The printed accuracies should land in family with the reported multi-seed
numbers (backprop ~97.8%, Hebbian ~89%) -- this is one seed, so expect a
little noise, but not a different ballpark. If it's wildly off, that's a
real bug; don't ship the resulting JSON.

IMPORTANT: the preprocessing (divide by 255 -> subtract the dataset mean
image -> L2-normalise) is exactly `common.load_dataset`'s preprocessing, and
is mirrored in `docs/js/net.js`'s `preprocess()`. If you ever change the
centering/normalization in `common.py`, update `docs/js/net.js` to match, or
the in-browser demo will quietly produce garbage.
"""

import base64
import json
import os
import sys

import numpy as np
import torch
import torchvision

sys.path.insert(0, os.path.dirname(__file__))
import common as C

# ---------------------------------------------------------------- config
SEED           = 0
WIDTH          = 400           # hidden units -- must match the rest of the repo
BP_EPOCHS      = 8              # matches common.run_seed_variability_experiment's default
HEB_EPOCHS     = 3              # matches common.hebbian_W1's default
READOUT_EPOCHS = 100            # matches common.train_readout's default

N_EXAMPLES     = 200            # real test digits shipped for the "Real test digit" button
OUT_PATH       = os.path.join("docs", "assets", "models.json")

device = C.device
print(f"device: {device}")


# ---------------------------------------------------------------- quantise
def q8(t):
    """int8 + a single scale per matrix. 4x smaller than float32, and the
    demo cannot tell the difference."""
    a = t.detach().cpu().numpy().astype(np.float32)
    scale = float(np.abs(a).max()) / 127.0
    if scale == 0:
        scale = 1e-8
    q = np.clip(np.round(a / scale), -127, 127).astype(np.int8)
    return {
        "shape": list(a.shape),
        "scale": scale,
        "data": base64.b64encode(q.tobytes()).decode("ascii"),
    }


def flist(t):
    return [round(float(v), 6) for v in t.detach().cpu().numpy().ravel()]


# ---------------------------------------------------------------- main
def main():
    print("loading MNIST...")
    X_train, y_train, X_test, y_test, raw_test, mean = C.load_dataset(torchvision.datasets.MNIST)

    # ---- backprop baseline: models/backprop.py's architecture, common.py's training loop
    print("\ntraining backprop baseline...")
    net = C.build_backprop_net(WIDTH, seed=SEED)
    net, _ = C.train_backprop(net, X_train, y_train, epochs=BP_EPOCHS)
    bp_acc = C.eval_acc(lambda x: net(x), X_test, y_test)
    print(f"  backprop test acc {bp_acc:.2f}%")

    # ---- Hebbian hidden layer: common.hebbian_W1's default mode="lateral"
    # (Foldiak anti-Hebbian lateral inhibition + Turrigiano homeostasis), no labels.
    print("\ntraining Hebbian hidden layer (no labels)...")
    Wh = C.hebbian_W1(X_train, WIDTH, epochs=HEB_EPOCHS, seed=SEED)  # HebbianState(W, L, theta)

    print("training the shared supervised readout on Hebbian features...")
    ro = C.build_readout(WIDTH, seed=SEED)
    H_train = C.hebbian_features(X_train, Wh)
    ro, _ = C.train_readout(ro, H_train, y_train, epochs=READOUT_EPOCHS)
    heb_acc = C.eval_acc(lambda x: ro(C.hebbian_features(x, Wh)), X_test, y_test)
    print(f"  hebbian test acc {heb_acc:.2f}%")

    print("\n" + "=" * 52)
    print(f"  backprop   {bp_acc:6.2f}%")
    print(f"  hebbian    {heb_acc:6.2f}%")
    print(f"  gap        {bp_acc - heb_acc:6.2f} points")
    print("=" * 52)
    print("These should be in family with the reported multi-seed numbers")
    print("(97.75% / 89.03%). If they are wildly off, something is wrong --")
    print("do not ship them.\n")

    # a handful of real test digits for the "Real test digit" button
    idx = torch.randperm(raw_test.shape[0])[:N_EXAMPLES]
    examples = [{
        "px": base64.b64encode(
            (raw_test[i].numpy() * 255).astype(np.uint8).tobytes()
        ).decode("ascii"),
        "y": int(y_test[i]),
    } for i in idx]

    payload = {
        "meta": {
            "arch": "784-400-10",
            "dataset": "MNIST",
            "seed": SEED,
            "note": "Weights for the in-browser demo, trained with models/common.py's own "
                    "functions. Headline numbers on the site are 5-seed means from the "
                    "repo's sweeps; these are one seed.",
        },
        "pre": {
            "mean_image": flist(mean.view(-1)),
            "l2_normalize": True,
        },
        "models": {
            "backprop": {
                "acc": round(bp_acc, 2),
                "W1": q8(net[0].weight), "b1": flist(net[0].bias),
                "W2": q8(net[2].weight), "b2": flist(net[2].bias),
            },
            "hebbian": {
                "acc": round(heb_acc, 2),
                "W1": q8(Wh.W),
                "theta": flist(Wh.theta),
                "L": q8(Wh.L),
                "settle_steps": 4,  # matches common.hebbian_features's default relax_iters
                "Wr": q8(ro.weight), "br": flist(ro.bias),
            },
        },
        "examples": examples,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f)

    mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"wrote {OUT_PATH}  ({mb:.2f} MB)")
    print("\nnow run:  python -m http.server --directory docs 8000")


if __name__ == "__main__":
    main()
