"""
semisl_det_v2.py - Wave 2-C v2: semi-supervised CenterNet on FashionLetters36 (Unbiased-Teacher style).

v2 (11-11) fixes the first version, whose unsupervised loss turned every object the teacher missed into a
background target (pseudo-labels 0.1-2 per image vs ~6.5 true objects) and lowered mAP below the labeled-only
baseline: pseudo threshold 0.5 -> 0.3; background is learned only where the teacher heat is < --tau-lo (0.1),
cells in between are ignored; unlabeled images without any pseudo box are left out of the unsupervised loss;
the loss is normalised by max(1, pseudo boxes); EMA 0.9996 -> 0.999 (15k updates, not 180k); pseudo-label
precision / recall against the unlabeled ground truth is logged (diagnostic only, never used for training).

Run from the CenterNet repository folder (it imports common/ and detectors/), env ssl_env:
    python semisl_det_v2.py --ratio p10 --init scratch --tag det2_p10_scratch
    python semisl_det_v2.py --ratio p10 --init ~/SSL_Wave1b/runs/ssl_barlow/backbone_e200.pth --tag det2_p10_barlow200
    python semisl_det_v2.py --ratio p01 --init scratch --tag smoke --smoke --device cpu

Data: ~/SSL_Wave2/roots/<ratio>/datasets/FashionLetters36 (make_label_splits.py). Labeled = its trainval list
(the labeled subset repeated to ~4,100 samples per epoch), unlabeled = its unlabeled.txt (the other train images;
the loader parses their XML but the targets are discarded), test = the 1,024 test images.
Protocol = the Wave 2-A label-efficiency runs (train_image_centernet_lab.py): same model, common init,
--init backbone, Adam 1e-4 constant, micro batch 2 x accum 4, the same number of optimizer updates
(epochs x ceil(labeled samples / 8)), common augmentation, PreciseBN before evaluation, seed 777.
Those runs are the "no unlabeled data" baseline of this script.

Semi-supervised part (Unbiased Teacher, Liu et al. 2021, adapted to CenterNet):
  burn-in   the first --burnin fraction of updates is supervised only; then teacher = copy of student
  teacher   EMA of the student (--ema 0.9996), no gradient. [Unified] BN uses batch statistics with momentum 0
            (running statistics are not used or updated during training; CenterNet needs PreciseBN for them)
  pseudo    teacher detections on the weak view (= the common augmentation) with score >= --tau, common NMS
  student   heat-map focal loss on the strong view of the same image vs the pseudo boxes, weight --lam
            (UT keeps only classification losses for pseudo-labels; CenterNet's size / offset losses are dropped);
            negatives only where the teacher heat (per class) < --tau-lo, images with no pseudo box skipped
  strong    on the GPU, geometry unchanged so the pseudo boxes stay valid: brightness / contrast x U(0.6, 1.4),
            Gaussian blur p 0.5, Gaussian noise p 0.5, random erasing (p 0.7 / 0.5 / 0.3, area 1-5 %, random value)
Final: PreciseBN (1,600 labeled samples) then test mAP50 of the teacher (primary) and of the student.
Writes ~/SSL_Wave2/runs/<tag>/result.json (+ teacher.pth, student.pth) and ~/SSL_Wave2/logs/<tag>.log ("DONE <tag>").
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ratio", required=True, choices=["p01", "p05", "p10", "p25"])
ap.add_argument("--init", default="scratch")
ap.add_argument("--tag", required=True)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--burnin", type=float, default=0.3)
ap.add_argument("--tau", type=float, default=0.3)
ap.add_argument("--tau-lo", type=float, default=0.1, help="teacher heat below this = background")
ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--ema", type=float, default=0.999)
ap.add_argument("--eval-every", type=int, default=5, help="epochs between teacher test evaluations (0 = final only)")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--device", default=None)
ap.add_argument("--smoke", action="store_true", help="1 epoch on 16 labeled samples, burn-in 0.5, 8 test images")
a = ap.parse_args()

HOME = Path.home()
os.environ["PHYSICAL_AI_ROOT"] = str(HOME / "SSL_Wave2" / "roots" / a.ratio)     # before importing common
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
import torch.nn as nn                                                 # noqa: E402
import torch.nn.functional as F                                       # noqa: E402

from common import config as C                                        # noqa: E402
from common.darknet import weights_init_normal                        # noqa: E402
from common.datasets.voc import VOCDetection                          # noqa: E402
from common.geometry import batched_nms_2d                            # noqa: E402
from torchvision.ops import box_iou                                   # noqa: E402
from common.precise_bn import recompute_bn_stats                      # noqa: E402
from common.trainer import make_loader                                # noqa: E402
from detectors.anchor_free.common import split_targets, to_raw        # noqa: E402
from detectors.centernet.build import DOWN_RATIO, _sigmoid           # noqa: E402
from tasks.image_task import ImageTask                                # noqa: E402
from train_image_centernet_ssl import CenterNetSSL                    # noqa: E402

P = C.PROFILE
IMG = 608
OUT = HOME / "SSL_Wave2"
run_dir, log_path = OUT / "runs" / a.tag, OUT / "logs" / f"{a.tag}.log"
run_dir.mkdir(parents=True, exist_ok=True)
log_path.parent.mkdir(parents=True, exist_ok=True)


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------------------------------------ strong aug
def strong(x):
    """x [B,3,H,W] in 0..1 -> photometric strong view (geometry unchanged)."""
    b, dev = x.shape[0], x.device
    v = lambda t: t.view(b, 1, 1, 1)                                               # noqa: E731
    u = lambda lo, hi: torch.rand(b, device=dev) * (hi - lo) + lo                   # noqa: E731
    x = x * v(u(0.6, 1.4))
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (m + (x - m) * v(u(0.6, 1.4))).clamp(0, 1)
    blur_on = v(torch.rand(b, device=dev) < 0.5)
    k = torch.tensor([1., 4., 6., 4., 1.], device=dev)
    k = (k[:, None] * k[None, :]) / 256.0
    blurred = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), k.expand(3, 1, 5, 5), groups=3)
    x = torch.where(blur_on, blurred, x)
    noise_on = v(torch.rand(b, device=dev) < 0.5)
    x = torch.where(noise_on, (x + torch.randn_like(x) * 0.03).clamp(0, 1), x)
    H, W = x.shape[-2:]
    for p in (0.7, 0.5, 0.3):                                                       # random erasing
        on = torch.rand(b, device=dev) < p
        area = u(0.01, 0.05) * H * W
        ratio = torch.exp(u(math.log(0.3), math.log(3.3)))
        hh = (area * ratio).sqrt().clamp(1, H - 1).long()
        ww = (area / ratio).sqrt().clamp(1, W - 1).long()
        y0 = (torch.rand(b, device=dev) * (H - hh)).long()
        x0 = (torch.rand(b, device=dev) * (W - ww)).long()
        ys, xs = torch.arange(H, device=dev).view(1, H, 1), torch.arange(W, device=dev).view(1, 1, W)
        box = ((ys >= y0.view(b, 1, 1)) & (ys < (y0 + hh).view(b, 1, 1)) &
               (xs >= x0.view(b, 1, 1)) & (xs < (x0 + ww).view(b, 1, 1)) & on.view(b, 1, 1))
        x = torch.where(box.unsqueeze(1), torch.rand(b, 1, 1, 1, device=dev).expand_as(x), x)
    return x


# ------------------------------------------------------------------------------------------------ pseudo labels
@torch.no_grad()
def pseudo_targets(raw, tau):
    """raw [B, M, 5 + C] (cx, cy, w, h input px, obj = 1, class probs after peak NMS) -> targets [N, 6]."""
    rows = []
    for b, pred in enumerate(raw):
        conf, cls = pred[:, 5:].max(1)
        keep = conf >= tau
        pred, conf, cls = pred[keep], conf[keep], cls[keep]
        if conf.numel() == 0:
            continue
        boxes = torch.stack([pred[:, 0] - pred[:, 2] / 2, pred[:, 1] - pred[:, 3] / 2,
                             pred[:, 0] + pred[:, 2] / 2, pred[:, 1] + pred[:, 3] / 2], 1)
        k = batched_nms_2d(boxes, conf, cls, P["nms_iou"])[: P["max_det"]]
        p = pred[k]
        t = torch.zeros((len(k), 6), device=raw.device)
        t[:, 0] = b
        t[:, 1] = cls[k].float()
        t[:, 2:6] = (p[:, :4] / IMG).clamp(0, 1)
        rows.append(t)
    return torch.cat(rows) if rows else torch.zeros((0, 6), device=raw.device)


@torch.no_grad()
def decode(out):
    """CenterNet.forward decode (same lines, no loss): net outputs -> raw [B, M, 5 + C] and the sigmoid heat."""
    B, _, H, W = out["hm"].shape
    heat = out["hm"].sigmoid()
    hmax = F.max_pool2d(heat, (3, 3), stride=1, padding=1)
    heat_nms = heat * (hmax == heat).float()
    ys, xs = torch.meshgrid(torch.arange(H, device=heat.device, dtype=heat.dtype),
                            torch.arange(W, device=heat.device, dtype=heat.dtype), indexing="ij")
    reg, wh = out["reg"], out["wh"]
    cx, cy = xs + reg[:, 0], ys + reg[:, 1]
    boxes = torch.stack([cx - wh[:, 0] / 2, cy - wh[:, 1] / 2, cx + wh[:, 0] / 2, cy + wh[:, 1] / 2], -1)
    boxes = boxes.reshape(B, -1, 4) * DOWN_RATIO
    raw = to_raw(boxes, heat_nms.permute(0, 2, 3, 1).reshape(B, -1, heat.shape[1]))
    return raw, heat


def unsup_hm_loss(logits, gt, ignore):
    """CenterNet focal loss with an ignore mask on the negatives, normalised by max(1, positives)."""
    pred = _sigmoid(logits.clone())
    pos = gt.eq(1).float()
    neg = gt.lt(1).float() * (~ignore).float()
    pos_loss = (torch.log(pred) * torch.pow(1 - pred, 2) * pos).sum()
    neg_loss = (torch.log(1 - pred) * torch.pow(pred, 2) * torch.pow(1 - gt, 4) * neg).sum()
    return -(pos_loss + neg_loss) / pos.sum().clamp_min(1.0)


@torch.no_grad()
def pseudo_quality(pt, gt, B):
    """Precision / recall of the pseudo boxes vs the unlabeled ground truth at IoU 0.5, same class (log only)."""
    ps, gs = split_targets(pt, B, IMG), split_targets(gt, B, IMG)
    tp_p = n_p = tp_g = n_g = 0
    for p, g in zip(ps, gs):
        n_p, n_g = n_p + len(p["labels"]), n_g + len(g["labels"])
        if len(p["labels"]) and len(g["labels"]):
            iou = box_iou(p["boxes"], g["boxes"])
            same = p["labels"].view(-1, 1) == g["labels"].view(1, -1)
            hit = (iou >= 0.5) & same
            tp_p += int(hit.any(1).sum())
            tp_g += int(hit.any(0).sum())
    return tp_p, n_p, tp_g, n_g


def teacher_mode(teacher):
    teacher.train()
    for m in teacher.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.momentum = 0.0                   # batch statistics, running statistics untouched


@torch.no_grad()
def ema_update(teacher, student, m):
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.mul_(m).add_(ps.detach(), alpha=1 - m)


def evaluate(model, lab_set, test_set, device, tag):
    m = copy.deepcopy(model)
    for mod in m.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.momentum = 0.1
    recompute_bn_stats(m, lab_set, device, workers=a.workers, seed=a.seed)
    eargs = SimpleNamespace(batch_size=2, workers=a.workers, seed=a.seed, run_name=tag, img_size=IMG)
    res = ImageTask().evaluate(m, test_set, eargs, device)
    return float(res["ap50"]["summary"]["mAP"]), m


# ------------------------------------------------------------------------------------------------ main
def main():
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    names = list(C.FL36_CLASSES)
    lab_set = VOCDetection(C.VOC_ROOT, "trainval", IMG, names, train=True, hflip_prob=P["hflip_prob"],
                           aug=P["image_aug"], max_samples=16 if a.smoke else None)
    unl_set = VOCDetection(C.VOC_ROOT, "unlabeled", IMG, names, train=True, hflip_prob=P["hflip_prob"],
                           aug=P["image_aug"])
    test_set = VOCDetection(C.VOC_ROOT, "test", IMG, names, train=False, max_samples=8 if a.smoke else None)
    epochs, burnin = (1, 0.5) if a.smoke else (a.epochs, a.burnin)
    mb, accum = 2, 4
    lab_loader = make_loader(lab_set, mb, a.workers, True, a.seed)
    unl_loader = make_loader(unl_set, mb, a.workers, True, a.seed + 1)
    updates_per_epoch = math.ceil(len(lab_loader) / accum)
    total_updates = epochs * updates_per_epoch
    burnin_updates = int(burnin * total_updates)

    student = CenterNetSSL(len(names)).to(device)
    student.apply(weights_init_normal)
    if a.init != "scratch":
        student.load_darknet_weights(str(Path(a.init).expanduser()))
    opt = torch.optim.Adam(student.parameters(), lr=P["lr"])
    teacher = None
    log(f"start {a.tag}: ratio {a.ratio} (root {C.VOC_ROOT}), labeled samples {len(lab_set)} "
        f"({len(set(lab_set.ids))} images), unlabeled images {len(unl_set)}, test {len(test_set)}, init {a.init}, "
        f"epochs {epochs}, updates {total_updates} (burn-in {burnin_updates}), tau {a.tau}, tau_lo {a.tau_lo}, lam {a.lam}, "
        f"ema {a.ema}, device {device}")

    unl_iter = iter(unl_loader)
    update, curve, t0 = 0, [], time.time()
    for epoch in range(1, epochs + 1):
        student.train()
        sums = {"sup": 0.0, "unsup": 0.0, "pseudo_per_img": 0.0, "n": 0, "nu": 0}
        pending = 0
        opt.zero_grad(set_to_none=True)
        for bi, (_, img_l, tgt_l) in enumerate(lab_loader, 1):
            img_l = img_l.to(device, dtype=torch.float32, non_blocking=True)
            tgt_l = tgt_l.to(device, dtype=torch.float32, non_blocking=True)
            loss_sup, _ = student(img_l, tgt_l)
            loss = loss_sup
            if teacher is not None:
                try:
                    _, img_u, gt_u = next(unl_iter)               # gt_u: logged pseudo-label quality only
                except StopIteration:
                    unl_iter = iter(unl_loader)
                    _, img_u, gt_u = next(unl_iter)
                img_u = img_u.to(device, dtype=torch.float32, non_blocking=True)
                Bu = img_u.shape[0]
                with torch.no_grad():
                    raw_t, heat_t = decode(teacher.net(img_u))
                    pt = pseudo_targets(raw_t, a.tau)
                q = pseudo_quality(pt, gt_u.to(device, dtype=torch.float32), Bu)
                for k_, v_ in zip(("tp_p", "n_p", "tp_g", "n_g"), q):
                    sums[k_] = sums.get(k_, 0) + v_
                has = torch.zeros(Bu, dtype=torch.bool, device=device)
                if len(pt):
                    has[pt[:, 0].long()] = True
                sums["pseudo_per_img"] += len(pt) / Bu
                sums["nu"] += 1
                if has.any():
                    keep_b = torch.nonzero(has).flatten()
                    remap = torch.full((Bu,), -1, dtype=torch.long, device=device)
                    remap[keep_b] = torch.arange(len(keep_b), device=device)
                    ptk = pt.clone()
                    ptk[:, 0] = remap[pt[:, 0].long()].float()
                    out = student.net(strong(img_u[keep_b]))
                    H, W = out["hm"].shape[-2:]
                    hm_t = student.build_targets(split_targets(ptk, len(keep_b), IMG), H, W, device)[0]
                    ignore = heat_t[keep_b] >= a.tau_lo
                    loss_unsup = unsup_hm_loss(out["hm"], hm_t, ignore)
                    loss = loss + a.lam * loss_unsup
                    sums["unsup"] += float(loss_unsup.detach())
            if not torch.isfinite(loss):
                log(f"  WARNING: non-finite loss at epoch {epoch} batch {bi}, skipped")
                continue
            (loss / accum).backward()
            sums["sup"] += float(loss_sup.detach())
            sums["n"] += 1
            pending += 1
            if pending == accum or bi == len(lab_loader):
                if all(p.grad is None or torch.isfinite(p.grad).all() for p in student.parameters()):
                    opt.step()
                opt.zero_grad(set_to_none=True)
                pending = 0
                update += 1
                if teacher is None and update >= burnin_updates:
                    teacher = copy.deepcopy(student)
                    for p in teacher.parameters():
                        p.requires_grad = False
                    teacher_mode(teacher)
                    log(f"burn-in done at update {update}: teacher = student, semi-supervised training starts")
                elif teacher is not None:
                    ema_update(teacher, student, a.ema)
        nu = max(sums["nu"], 1)
        prec = sums.get("tp_p", 0) / max(sums.get("n_p", 0), 1)
        rec = sums.get("tp_g", 0) / max(sums.get("n_g", 0), 1)
        log(f"epoch {epoch}/{epochs} update {update}/{total_updates} sup {sums['sup'] / max(sums['n'], 1):.4f} "
            f"unsup {sums['unsup'] / nu:.4f} pseudo/img {sums['pseudo_per_img'] / nu:.2f} "
            f"pseudo precision {prec:.3f} recall {rec:.3f} ({(time.time() - t0) / 60:.1f} min)")
        if sums["nu"]:
            curve.append({"epoch": epoch, "pseudo_per_img": sums["pseudo_per_img"] / nu,
                          "pseudo_precision": prec, "pseudo_recall": rec})
        if (a.eval_every and teacher is not None and epoch % a.eval_every == 0 and epoch < epochs):
            m, _ = evaluate(teacher, lab_set, test_set, device, a.tag)
            curve.append({"epoch": epoch, "teacher_map50": m})
            log(f"eval epoch {epoch}: teacher test mAP50 {m:.4f}")
            teacher_mode(teacher)
    if teacher is None:                                   # burn-in never finished (smoke edge case)
        teacher = copy.deepcopy(student)
    t_map, t_eval = evaluate(teacher, lab_set, test_set, device, a.tag)
    s_map, s_eval = evaluate(student, lab_set, test_set, device, a.tag)
    torch.save(t_eval.state_dict(), run_dir / "teacher.pth")
    torch.save(s_eval.state_dict(), run_dir / "student.pth")
    res = {"tag": a.tag, "ratio": a.ratio, "init": a.init, "epochs": epochs, "updates": total_updates,
           "burnin_updates": burnin_updates, "tau": a.tau, "tau_lo": a.tau_lo, "lam": a.lam, "ema": a.ema,
           "version": 2,
           "teacher_map50": t_map, "student_map50": s_map, "curve": curve,
           "minutes": round((time.time() - t0) / 60, 1)}
    (run_dir / "result.json").write_text(json.dumps(res, indent=1))
    log(f"final test mAP50 teacher {t_map:.4f} student {s_map:.4f}")
    log(f"DONE {a.tag}")


if __name__ == "__main__":
    main()
