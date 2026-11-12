# SemiSL_Wave2_FL36

**Semi-supervised learning on FashionLetters36 - PAI-SG SemiSL Wave 2**

Part of the *Physical AI* reference series of the **Statera Working Group**. With 1 / 5 / 10 % of the labels, how much
do the unlabeled images add - for classification and for detection - on top of, or instead of, self-supervised
pre-training ([SSL_Wave1_FL36](https://github.com/Statera-Workshop/SSL_Wave1_FL36))? All runs: RTX 4070 Laptop 8 GB.

## Labeled splits

Nested image subsets of the 4,096 FL36 train images, seed 777 (`code/make_label_splits.py`, lists in `splits/`):
1 % = 41 images / 258 objects, 5 % = 205 / 1,340, 10 % = 410 / 2,653; all 36 classes present in every split.
Classification uses the object crops of the labeled images (same split); the unlabeled set is all train data with
labels never read.

## A. Crop36 classification (`code/semisl_cls.py`)

Supervised-only, Pseudo-Label, Mean Teacher, FixMatch, FlexMatch with the Microsoft USB classic_cv defaults
(SGD 0.03 nesterov, wd 5e-4, batch 64, cosine 7/16, EMA 0.999 evaluated, threshold 0.95, FixMatch uratio 7),
10k steps, ResNet-18 from scratch or from VICReg 200 (SSL). GPU augmentation; no horizontal flip.
[Unified]: Mean Teacher uses two weak views (paper); FixMatch's weak unlabeled view has its own no-grad forward.

EMA test top-1 (full table with macro F1: `results/cls_summary.md`):

| Method | 1 % scratch | 1 % VICReg | 5 % scratch | 5 % VICReg | 10 % scratch | 10 % VICReg |
| --- | --- | --- | --- | --- | --- | --- |
| supervised only | 0.502 | 0.780 | 0.833 | 0.892 | 0.873 | 0.908 |
| Pseudo-Label | 0.743 | 0.821 | 0.867 | 0.901 | 0.874 | 0.910 |
| Mean Teacher | 0.685 | 0.702 | 0.877 | 0.889 | 0.896 | 0.906 |
| FixMatch | 0.860 | 0.797 | 0.911 | 0.908 | 0.919 | 0.916 |
| FlexMatch | 0.868 | 0.843 | 0.912 | 0.913 | 0.913 | 0.918 |

100 % labels, supervised: 0.926 (scratch) / 0.932 (VICReg).
FixMatch / FlexMatch come within 1 point of full supervision with 5-10 % labels; at 1 % FlexMatch reaches 0.868
vs 0.502 supervised-only. SSL initialisation helps the simple methods most; for FixMatch at 1 % it hurts
(0.860 -> 0.797; confident wrong pseudo-labels early - a hypothesis, pseudo-label accuracy was not logged).

## B. Semi-supervised CenterNet (`code/semisl_det_v2.py`)

Unbiased-Teacher style on CenterNet: burn-in on labeled data, EMA teacher, pseudo boxes from the teacher on the weak
view, heat-map focal loss on a photometric strong view; same updates, protocol and test set as the labeled-only
CenterNet runs of SSL_Wave1_FL36 (the baseline). Full table: `results/det_summary.md`.

**Version 1 failed** (`code/semisl_det.py`, kept as a record): with threshold 0.5 only 0.1-2 pseudo boxes per image
were produced (true objects ~6.5), every missed object became a background target, and mAP fell far below the
labeled-only baseline (10 % scratch 0.564 -> 0.21). **Version 2**: threshold 0.3, background only where the teacher
heat is < 0.1 (cells in between ignored), images without pseudo boxes skipped, EMA 0.999; pseudo-label precision /
recall vs the unlabeled ground truth logged (never trained on).

Teacher mAP50, labeled-only -> version 2 (last-epoch pseudo-label precision):

| Labels | from scratch | Barlow 200 (SSL) |
| --- | --- | --- |
| 1 % | 0.139 -> 0.145 (0.21) | 0.319 -> 0.319 (0.34) |
| 5 % | 0.463 -> 0.414 (0.48) | 0.676 -> 0.687 (0.63) |
| 10 % | 0.564 -> 0.378 (0.46) | 0.771 -> 0.778 (0.73) |

Burn-in 60 % instead of 30 % (10 %, scratch; stronger teacher before pseudo-labelling): teacher 0.5705,
student 0.5617; the labeled-only baseline is 0.564, so a stronger teacher removes the loss but adds no gain.

**Conclusions.** Pseudo-label precision decides the outcome: with a weak teacher (from scratch, precision ~0.46)
semi-supervised training hurts; on a good SSL backbone it no longer hurts but adds nothing beyond run-to-run
variation within the 30-epoch budget. In this setting a good SSL backbone (SSL_Wave1_FL36) gives far more than the
unlabeled-data term: Barlow 200 with 10 % labels alone (0.771) beats from-scratch with 100 % labels (0.611).

## Reproduce

Ubuntu, `code/queue_runner.sh` with the jobs in `code/ubuntu_jobs.txt` (Wave 2 sections). Data and SSL backbones:
SSL_Wave1_FL36 (Releases). Python 3.10, PyTorch 2.5.1 + cu124, torchvision 0.20.1, scikit-learn 1.7.

Third-party code: [THIRD_PARTY.md](THIRD_PARTY.md).
