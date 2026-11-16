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

Teacher mAP50 at two budgets. Labeled-only = the same CenterNet without the unlabeled term (SSL_Wave1_FL36).
30 epochs: burn-in 30 %; 90 epochs: burn-in 60 %.

| Labels | Init | labeled-only 30 ep | + SemiSL 30 ep | labeled-only 90 ep | + SemiSL 90 ep |
| --- | --- | --- | --- | --- | --- |
| 1 % | from scratch | 0.139 | 0.145 | 0.131 | - |
| 1 % | Barlow 200 | 0.319 | 0.319 | 0.330 | 0.348 |
| 5 % | from scratch | 0.463 | 0.414 | 0.605 | **0.725** |
| 5 % | Barlow 200 | 0.676 | 0.687 | 0.711 | **0.773** |
| 10 % | from scratch | 0.564 | 0.378 (burn-in 60 %: 0.571) | 0.725 / 0.720 | **0.793 / 0.801** |
| 10 % | Barlow 200 | 0.771 | 0.778 | 0.770 / 0.774 / 0.775 | **0.821** |

Last-epoch pseudo-label precision at 30 epochs: from scratch 0.21-0.48, Barlow 200 0.34-0.73 (all runs and
pseudo-label statistics: `results/det_summary.md`).

**Conclusions.**

1. **The budget decides.** At 30 epochs the unlabeled term adds nothing on a good backbone and hurts with a weak
   teacher (10 % scratch 0.564 -> 0.378; a later burn-in only removes the loss). At 90 epochs with burn-in 60 % it
   adds **+0.06 to +0.12** (10 % scratch +0.075, reproduced with a second seed: 0.793 / 0.801 vs 0.725 / 0.720).
2. **SSL and semi-supervised learning add up.** Barlow 200 + SemiSL with 10 % labels reaches 0.821, against
   0.722 for from-scratch with the same labels and 0.860 for from-scratch with 100 % labels (90 epochs).
3. **Pseudo-label precision is the limit.** With 1 % labels (41 images) the teacher stays weak and the gain is
   small (+0.018 on Barlow 200).
4. The first conclusion of this repository (no gain from semi-supervised detection) was an artefact of the short
   budget, like the convergence effect of SSL in SSL_Wave1_FL36.

## Reproduce

Ubuntu, `code/queue_runner.sh` with the jobs in `code/ubuntu_jobs.txt` (Wave 2 sections). Data and SSL backbones:
SSL_Wave1_FL36 (Releases). Python 3.10, PyTorch 2.5.1 + cu124, torchvision 0.20.1, scikit-learn 1.7.

Third-party code: [THIRD_PARTY.md](THIRD_PARTY.md).
