# Semi-supervised CenterNet - test mAP50 (RTX 4070)

Version 1 = first implementation (tau 0.5, every non-pseudo cell is background, EMA 0.9996) - failed;
version 2 = tau 0.3, background only where teacher heat < 0.1, images without pseudo boxes skipped, EMA 0.999.
Pseudo-label precision / recall are measured against the unlabeled ground truth (logged only, never trained on).

| run | version | teacher mAP50 | student mAP50 | burn-in | tau | last pseudo/img | precision | recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| det2_p01_barlow200 | 2 | 0.3192 | 0.3131 | 4617/15390 | 0.3 | 7.08 | 0.339 | 0.395 |
| det2_p01_barlow200_e90 | 2 | 0.3477 | 0.3303 | 27702/46170 | 0.3 | 9.82 | 0.297 | 0.486 |
| det2_p01_scratch | 2 | 0.1445 | 0.1384 | 4617/15390 | 0.3 | 6.91 | 0.212 | 0.241 |
| det2_p05_barlow200 | 2 | 0.6872 | 0.6752 | 4617/15390 | 0.3 | 6.57 | 0.627 | 0.676 |
| det2_p05_barlow200_e90 | 2 | 0.7732 | 0.7613 | 27702/46170 | 0.3 | 6.72 | 0.703 | 0.780 |
| det2_p05_scratch | 2 | 0.4143 | 0.4009 | 4617/15390 | 0.3 | 6.25 | 0.479 | 0.491 |
| det2_p05_scratch_e90 | 2 | 0.7246 | 0.7135 | 27702/46170 | 0.3 | 6.49 | 0.688 | 0.737 |
| det2_p10_barlow200 | 2 | 0.7779 | 0.7629 | 4617/15390 | 0.3 | 6.32 | 0.728 | 0.758 |
| det2_p10_barlow200_e90 | 2 | 0.8207 | 0.8130 | 27702/46170 | 0.3 | 6.42 | 0.789 | 0.830 |
| det2_p10_scratch | 2 | 0.3778 | 0.3820 | 4617/15390 | 0.3 | 5.99 | 0.464 | 0.458 |
| det2_p10_scratch_bi60 | 2 | 0.5705 | 0.5617 | 9234/15390 | 0.3 | 6.19 | 0.601 | 0.611 |
| det2_p10_scratch_e90 | 2 | 0.7934 | 0.7726 | 27702/46170 | 0.3 | 6.26 | 0.777 | 0.797 |
| det2_p10_scratch_e90_s778 | 2 | 0.8010 | 0.7809 | 27702/46170 | 0.3 | 6.26 | 0.779 | 0.800 |
| det_p01_ut_barlow200 | 1 | 0.2926 | 0.3184 | 4617/15390 | 0.5 | nan | nan | nan |
| det_p01_ut_scratch | 1 | 0.0933 | 0.0961 | 4617/15390 | 0.5 | nan | nan | nan |
| det_p05_ut_barlow200 | 1 | 0.5412 | 0.5744 | 4617/15390 | 0.5 | nan | nan | nan |
| det_p05_ut_scratch | 1 | 0.2458 | 0.2602 | 4617/15390 | 0.5 | nan | nan | nan |
| det_p10_ut_barlow200 | 1 | 0.5886 | 0.6083 | 4617/15390 | 0.5 | nan | nan | nan |
| det_p10_ut_scratch | 1 | 0.2099 | 0.2783 | 4617/15390 | 0.5 | nan | nan | nan |
