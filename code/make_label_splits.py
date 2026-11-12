"""
make_label_splits.py - labeled subsets of FashionLetters36 for Wave 2 (label efficiency / semi-supervised).

    python make_label_splits.py            # source: ~/SSL_Wave1/datasets/FashionLetters36 (symlink to the original)

For each ratio r in 1 / 5 / 10 / 25 % it builds a dataset root
    ~/SSL_Wave2/roots/p01/datasets/FashionLetters36/
        JPEGImages  -> original (symlink)          Annotations -> original (symlink)
        ImageSets/Main/trainval.txt   the labeled subset, each id repeated k times (k = round(4096 / n))
        ImageSets/Main/test.txt       the original 1,024 test ids
        ImageSets/Main/labeled.txt    the n labeled ids (once)      unlabeled.txt   the other train ids
Subsets are drawn with seed 777 and are NESTED (1 % in 5 % in 10 % in 25 %). Repeating the list makes every
ratio take ~4,100 samples per epoch, so 30 epochs = the same number of optimizer steps as the full-data run:
only the number of labeled images differs. Use with  PHYSICAL_AI_ROOT=~/SSL_Wave2/roots/p01 .
Writes ~/SSL_Wave2/label_splits.json (counts, objects and per-class objects per ratio). Prints RESULT OK / FAIL.
"""
import json
import os
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

SRC = Path(os.path.expanduser("~/SSL_Wave1/datasets/FashionLetters36")).resolve()
OUT = Path(os.path.expanduser("~/SSL_Wave2"))
RATIOS = [("p01", 0.01), ("p05", 0.05), ("p10", 0.10), ("p25", 0.25)]
TARGET = 4096

ids = [x for x in (SRC / "ImageSets/Main/trainval.txt").read_text().split() if x]
test = (SRC / "ImageSets/Main/test.txt").read_text()
assert len(ids) == 4096, f"expected 4096 train ids, found {len(ids)}"
order = ids[:]
random.Random(777).shuffle(order)          # nested subsets = prefixes of one permutation


def objects(sub):
    c = Counter()
    for i in sub:
        for o in ET.parse(SRC / "Annotations" / f"{i}.xml").getroot().iter("object"):
            c[o.find("name").text.strip().lower()] += 1
    return c


report, fails = {"source": str(SRC), "seed": 777, "target_samples_per_epoch": TARGET}, []
prev = set()
for tag, r in RATIOS:
    n = round(r * len(ids))
    lab = order[:n]
    if not prev <= set(lab):
        fails.append(f"{tag}: not nested")
    prev = set(lab)
    k = max(1, round(TARGET / n))
    root = OUT / "roots" / tag / "datasets" / "FashionLetters36"
    m = root / "ImageSets" / "Main"
    m.mkdir(parents=True, exist_ok=True)
    for d in ("JPEGImages", "Annotations"):
        link = root / d
        if not link.exists():
            link.symlink_to(SRC / d)
    (m / "trainval.txt").write_text("\n".join(lab * k) + "\n")
    (m / "labeled.txt").write_text("\n".join(lab) + "\n")
    (m / "unlabeled.txt").write_text("\n".join(i for i in ids if i not in prev) + "\n")
    (m / "test.txt").write_text(test)
    c = objects(lab)
    present = sum(1 for v in c.values() if v > 0)
    report[tag] = {"ratio": r, "labeled_images": n, "repeat": k, "samples_per_epoch": n * k,
                   "objects": sum(c.values()), "classes_present": present, "per_class": dict(sorted(c.items()))}
    print(f"{tag}: {n:5d} labeled images x{k:3d} = {n * k} samples/epoch, {sum(c.values()):6d} objects, "
          f"classes {present}/36, min per class {min(c.values()) if present == 36 else 0}")
    if present < 36:
        print(f"   note: {36 - present} classes have no labeled object at {tag}")

report["result"] = "OK" if not fails else "FAIL"
(OUT / "label_splits.json").write_text(json.dumps(report, indent=1))
for f in fails:
    print("FAIL:", f)
print(f"RESULT {report['result']}  ({OUT / 'label_splits.json'})")
