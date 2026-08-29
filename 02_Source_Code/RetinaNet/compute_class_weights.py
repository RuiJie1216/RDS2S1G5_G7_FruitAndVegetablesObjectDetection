"""
compute_class_weights.py
==========================
Scans TRAIN_LABELS_DIR, counts boxes per class, and saves a
class_weights.pt tensor (length NUM_CLASSES, ordered same as
CLASS_NAMES) for model.py to load.

CHANGES (post class_diagnostic.py run on model_last):
    class_diagnostic.py --run model_last classified every class into
    buckets based on production-threshold recall vs near-zero-threshold
    ("ceiling") recall:

        LEARNING_FAIL - recall stays ~0 even at threshold 0.02. The
                         model isn't producing confident-ish boxes for
                         these at all. Loss re-weighting is unlikely to
                         fix this on its own (25 classes -- left alone
                         here, needs a different investigation).

        CALIBRATION   - recall is low at the production threshold
                         (0.35) but jumps substantially at threshold
                         0.02. The model DID learn these classes; their
                         confidence scores are just too low to clear
                         the production threshold. This is the bucket
                         loss re-weighting can plausibly help with.

    The 17 classes below are exactly the CALIBRATION bucket from that
    run. They get an EXTRA multiplier on top of the existing log-based
    weight, specifically to push their confidence scores up during
    training. LEARNING_FAIL / OTHER_LOW / CONFUSED / HEALTHY classes
    are untouched -- boosting a class that isn't a calibration problem
    doesn't address its actual failure mode and just adds noise.

    EXTRA_WEIGHT_MULTIPLIER is a blunt, single knob. Start at 2.0x and
    re-run class_diagnostic.py after training to see whether the
    CALIBRATION classes' prod_recall moved -- adjust from there rather
    than assuming this value is correct on the first try.

Usage:
    python compute_class_weights.py
"""

import glob
import os
import torch

from config import CLASS_NAMES, NUM_CLASSES, TRAIN_LABELS_DIR

OUT_PATH = "class_weights.pt"

# ---------------------------------------------------------------------------
# CALIBRATION-bucket classes from class_diagnostic.py (run on model_last,
# prod_score=0.35, low_score=0.02, iou=0.5). Recall was low at the
# production threshold but recovered substantially at a near-zero
# threshold -- meaning the model has learned these classes but their
# scores aren't clearing SCORE_THRESHOLD reliably.
# ---------------------------------------------------------------------------
CALIBRATION_CLASSES = {
    "eggplant/aubergine",
    "pineapple",
    "bell pepper/capsicum",
    "lettuce",
    "watermelon",
    "mushroom",
    "onion",
    "potato",
    "almond",
    "strawberry",
    "pea/pea food",
    "tomato",
    "carrot",
    "banana",
    "apple",
    "orange/orange fruit",
    "broccoli",
}

# How much extra weight to stack on top of the base log-weight for the
# classes above. This is deliberately a single global knob rather than
# a per-class value -- start here, re-run the diagnostic after training,
# and adjust based on which of the 17 actually moved.
EXTRA_WEIGHT_MULTIPLIER = 2.0


def read_class_ids(path):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                ids.append(int(parts[0]))
            except ValueError:
                continue
    return ids


def main():
    counts = torch.zeros(NUM_CLASSES)

    for label_path in glob.glob(os.path.join(TRAIN_LABELS_DIR, "*.txt")):
        for cid in read_class_ids(label_path):
            if cid < NUM_CLASSES:
                counts[cid] += 1

    # avoid log(0) / div-by-0 for classes with 0 boxes
    weights = 1.0 / torch.log(counts + 10.0)

    # ------------------------------------------------------------------
    # Apply the targeted CALIBRATION-class boost BEFORE normalizing, so
    # the final mean-normalized weights stay on a sane overall scale
    # relative to the rest of the loss.
    # ------------------------------------------------------------------
    boosted = []
    for name in CLASS_NAMES:
        if name in CALIBRATION_CLASSES:
            boosted.append(name)

    for name in boosted:
        idx = CLASS_NAMES.index(name)
        weights[idx] *= EXTRA_WEIGHT_MULTIPLIER

    missing = CALIBRATION_CLASSES - set(boosted)
    if missing:
        print(f"[compute] WARNING: these CALIBRATION class names were not "
              f"found in CLASS_NAMES (check spelling/config.py): {missing}")

    weights = weights / weights.mean()  # normalize so total loss scale stays sane

    print(f"{'class':40s} {'n_boxes':>8s} {'weight':>8s}  {'boosted?':>8s}")
    print("-" * 70)
    for name, c, w in sorted(zip(CLASS_NAMES, counts.tolist(), weights.tolist()), key=lambda t: t[1]):
        tag = "YES" if name in CALIBRATION_CLASSES else ""
        print(f"{name:40s} {int(c):8d} {w:8.3f}  {tag:>8s}")

    torch.save(weights, OUT_PATH)
    print(f"\n[compute] Saved weights -> {OUT_PATH}")
    print(f"[compute] Boosted {len(boosted)}/{len(CALIBRATION_CLASSES)} CALIBRATION classes by {EXTRA_WEIGHT_MULTIPLIER}x")


if __name__ == "__main__":
    main()