"""
semisl_cls.py - Wave 2-B: semi-supervised classification on Crop36 (FashionLetters36 object crops, 36 classes).

Labeled split = the crops of the labeled IMAGES of ~/SSL_Wave2/roots/<ratio>/.../labeled.txt (same image-level
split as the detection runs); unlabeled = all 26,716 train crops, labels never read (USB convention: the
unlabeled set includes the labeled images). Test = the 6,699 Crop36 test crops.

Methods (Microsoft USB classic_cv defaults): SGD lr 0.03, momentum 0.9, nesterov, wd 5e-4 (not on BN / bias),
cosine lr * cos(7 pi k / 16 K), labeled batch 64, EMA 0.999 model evaluated, fp32.
  supervised   labeled CE only (weak aug)
  pseudolabel  + CE on argmax of the model's own weak-view prediction if max prob >= 0.95; uratio 1;
               weight ramps linearly to 1 over the first 40 % of steps
  meanteacher  + MSE(softmax(student(weak view 2)), softmax(EMA teacher(weak view 1))) x 50, uratio 1, same ramp
               [Unified] two independent weak views as in the paper (USB feeds a strong view)
  fixmatch     + CE(strong view, argmax(weak view)) where max prob >= 0.95; uratio 7
               [Unified] the weak unlabeled view goes through its own no-grad forward (BN batch statistics are
               not shared with the labeled / strong batch) so that 64 + 448 + 448 images fit in 8 GB
  flexmatch    fixmatch with class-wise thresholds 0.95 * M(beta_c), M(x) = x / (2 - x), with threshold warm-up
Augmentation on the GPU. weak: random translation +-12.5 % (no flip: letters are not mirror-invariant).
strong: weak + RandAugment (2 of: identity, autocontrast, brightness, contrast, sharpness, posterize, solarize,
rotate, shear x / y, translate x / y; random magnitude; equalize omitted) + Cutout (side up to 0.5, gray fill).
Input like the rest of Wave 1: grayscale x / 255 repeated to 3 channels, no mean / std.

    python semisl_cls.py --method fixmatch --ratio p10 --init scratch --tag cls_p10_fixmatch_scratch
    python semisl_cls.py --method fixmatch --ratio p10 --init ~/SSL_Wave1b/runs/ssl_vicreg/backbone_e200.pth \
        --tag cls_p10_fixmatch_vicreg200
    python semisl_cls.py --method fixmatch --ratio p01 --init scratch --tag smoke --smoke     # CPU-friendly check
Writes ~/SSL_Wave2/runs/<tag>/result.json and ~/SSL_Wave2/logs/<tag>.log (last line "DONE <tag>").
"""
import argparse
import copy
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

HOME = Path.home()
DATA = HOME / "SSL_Wave1" / "datasets" / "ssl"
FL36 = HOME / "SSL_Wave1" / "datasets" / "FashionLetters36"
ROOTS = HOME / "SSL_Wave2" / "roots"
OUT = HOME / "SSL_Wave2"
CLASSES = ("tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle_boot",
           *[chr(ord("A") + i) for i in range(26)])


# ------------------------------------------------------------------------------------------------ data
def load_data(ratio, device, max_test=None):
    tr, te = np.load(DATA / "crop36_train.npz"), np.load(DATA / "crop36_test.npz")
    ids = [x for x in (FL36 / "ImageSets/Main/trainval.txt").read_text().split() if x]
    cls_id = {c.lower(): i for i, c in enumerate(CLASSES)}
    img_of_crop, names = [], []
    for k, i in enumerate(ids):                      # same order as prep_ssl_data.py
        for o in ET.parse(FL36 / "Annotations" / f"{i}.xml").getroot().iter("object"):
            img_of_crop.append(k)
            names.append(cls_id[o.find("name").text.strip().lower()])
    img_of_crop = np.array(img_of_crop)
    if len(names) != len(tr["y"]) or not np.array_equal(np.array(names), tr["y"]):
        raise SystemExit("crop order does not match crop36_train.npz - rebuild with prep_ssl_data.py")
    if ratio == "p100":
        lab_mask = np.ones(len(names), bool)
    else:
        lab_ids = set((ROOTS / ratio / "datasets/FashionLetters36/ImageSets/Main/labeled.txt").read_text().split())
        lab_imgs = np.array([i in lab_ids for i in ids])
        lab_mask = lab_imgs[img_of_crop]
    xt, yt = te["x"], te["y"]
    if max_test:
        xt, yt = xt[:max_test], yt[:max_test]
    g = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)     # noqa: E731
    return dict(x=g(tr["x"]), y=g(tr["y"]), lab=g(np.nonzero(lab_mask)[0]), xt=g(xt), yt=yt,
                n_lab_imgs=int(len(set(img_of_crop[lab_mask]))), n_lab=int(lab_mask.sum()))


def to_float(xu8):                         # uint8 [B,H,W] -> float [B,1,H,W] in 0..1
    return xu8.float().div_(255.0).unsqueeze(1)


def to_input(x):                           # [B,1,H,W] -> [B,3,H,W]
    return x.expand(-1, 3, -1, -1)


# ------------------------------------------------------------------------------------------------ GPU aug
def affine(x, rot=None, shx=None, shy=None, tx=None, ty=None, fill=0.0):
    b = x.shape[0]
    z = torch.zeros(b, device=x.device)
    rot = z if rot is None else rot
    shx = z if shx is None else shx
    shy = z if shy is None else shy
    tx = z if tx is None else tx
    ty = z if ty is None else ty
    c, s = torch.cos(rot), torch.sin(rot)
    a = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)          # rotation
    sh = torch.stack([torch.stack([torch.ones_like(z), shx], -1), torch.stack([shy, torch.ones_like(z)], -1)], -2)
    m = torch.cat([a @ sh, torch.stack([tx, ty], -1).unsqueeze(-1) * 2], -1)            # translate in [-1,1] units
    grid = F.affine_grid(m, x.shape, align_corners=False)
    out = F.grid_sample(x - fill, grid, align_corners=False, padding_mode="zeros") + fill
    return out


def weak(x):
    b = x.shape[0]
    t = lambda: (torch.rand(b, device=x.device) * 2 - 1) * 0.125     # noqa: E731
    return affine(x, tx=t(), ty=t())


def _u(b, lo, hi, dev):
    return torch.rand(b, device=dev) * (hi - lo) + lo


def _sign(b, dev):
    return torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)


def strong(x):
    x = weak(x)
    b, dev = x.shape[0], x.device
    n_ops = 12
    ops = torch.randint(0, n_ops, (b, 2), device=dev)
    v = lambda t: t.view(b, 1, 1, 1)                                    # noqa: E731
    for j in range(1, n_ops):                                          # 0 = identity
        m = ((ops[:, 0] == j) | (ops[:, 1] == j))
        if not m.any():
            continue
        if j == 1:                                                     # autocontrast
            lo = x.amin(dim=(1, 2, 3), keepdim=True)
            hi = x.amax(dim=(1, 2, 3), keepdim=True)
            y = (x - lo) / (hi - lo).clamp_min(1e-3)
        elif j == 2:                                                   # brightness factor 0.05-0.95
            y = x * v(_u(b, 0.05, 0.95, dev))
        elif j == 3:                                                   # contrast
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            y = mean + (x - mean) * v(_u(b, 0.05, 0.95, dev))
        elif j == 4:                                                   # sharpness (blend with 3x3 smooth)
            k = torch.tensor([[1., 1., 1.], [1., 5., 1.], [1., 1., 1.]], device=dev).view(1, 1, 3, 3) / 13
            blur = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), k)
            y = blur + (x - blur) * v(_u(b, 0.05, 0.95, dev))
        elif j == 5:                                                   # posterize 4-8 bits
            bits = torch.randint(4, 9, (b,), device=dev).float()
            q = v(2.0 ** (8 - bits))
            y = torch.floor(x * 255 / q) * q / 255
        elif j == 6:                                                   # solarize
            th = v(torch.rand(b, device=dev))
            y = torch.where(x >= th, 1 - x, x)
        elif j == 7:                                                   # rotate +-30 deg
            y = affine(x, rot=_u(b, 0, 30, dev) * _sign(b, dev) * math.pi / 180)
        elif j == 8:
            y = affine(x, shx=_u(b, 0, 0.3, dev) * _sign(b, dev))
        elif j == 9:
            y = affine(x, shy=_u(b, 0, 0.3, dev) * _sign(b, dev))
        elif j == 10:
            y = affine(x, tx=_u(b, 0, 0.3, dev) * _sign(b, dev))
        else:                                                          # 11: translate y
            y = affine(x, ty=_u(b, 0, 0.3, dev) * _sign(b, dev))
        x = torch.where(v(m), y.clamp(0, 1), x)
    # cutout: square side U(0, 0.5) * size, gray fill
    h = x.shape[-1]
    side = (torch.rand(b, device=dev) * 0.5 * h).long()
    cy, cx = torch.randint(0, h, (b,), device=dev), torch.randint(0, h, (b,), device=dev)
    ar = torch.arange(h, device=dev)
    my = ((ar.view(1, h) - cy.view(b, 1)).abs() <= side.view(b, 1) // 2)
    mx = ((ar.view(1, h) - cx.view(b, 1)).abs() <= side.view(b, 1) // 2)
    cut = (my.unsqueeze(2) & mx.unsqueeze(1)).unsqueeze(1) & (side.view(b, 1, 1, 1) > 0)
    return torch.where(cut, torch.full_like(x, 0.5), x)


# ------------------------------------------------------------------------------------------------ model / optim
def make_model(init, device):
    m = torchvision.models.resnet18(weights=None)
    m.fc = nn.Identity()
    if init != "scratch":
        m.load_state_dict(torch.load(Path(init).expanduser(), map_location="cpu", weights_only=True))   # strict
    m.fc = nn.Linear(512, len(CLASSES))
    return m.to(device)


def make_optimizer(model, lr, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)                 # BN weights / biases: no wd
    return torch.optim.SGD([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
                           lr=lr, momentum=0.9, nesterov=True)


@torch.no_grad()
def ema_update(ema, model, m):
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.mul_(m).add_(p.detach(), alpha=1 - m)
    for be, b in zip(ema.buffers(), model.buffers()):
        be.copy_(b)


@torch.no_grad()
def evaluate(model, d, bs=1024):
    from sklearn.metrics import f1_score
    model.eval()
    pred = torch.cat([model(to_input(to_float(d["xt"][k:k + bs]))).argmax(1).cpu()
                      for k in range(0, len(d["xt"]), bs)]).numpy()
    model.train()
    return float((pred == d["yt"]).mean()), float(f1_score(d["yt"], pred, average="macro"))


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=["supervised", "pseudolabel", "meanteacher", "fixmatch",
                                                         "flexmatch"])
    ap.add_argument("--ratio", required=True, choices=["p01", "p05", "p10", "p25", "p100"])
    ap.add_argument("--init", default="scratch", help="scratch or an SSL backbone .pth (torchvision keys)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true", help="5 steps, batch 8, 512 test crops (works on CPU)")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.batch_size, a.eval_every = 5, 8, 0
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    run_dir, log_path = OUT / "runs" / a.tag, OUT / "logs" / f"{a.tag}.log"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    d = load_data(a.ratio, device, max_test=512 if a.smoke else None)
    uratio = {"supervised": 0, "pseudolabel": 1, "meanteacher": 1, "fixmatch": 7, "flexmatch": 7}[a.method]
    B, BU, tau = a.batch_size, a.batch_size * uratio, 0.95
    model = make_model(a.init, device)
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad = False
    opt = make_optimizer(model, 0.03, 5e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: max(0.0, math.cos(7 * math.pi * k / (16 * a.steps))))
    n_u = len(d["y"])
    selected = torch.full((n_u,), -1, dtype=torch.long, device=device)      # FlexMatch learning status
    log(f"start {a.tag}: method {a.method}, ratio {a.ratio} ({d['n_lab_imgs']} labeled images, {d['n_lab']} labeled "
        f"crops, {n_u} unlabeled crops), init {a.init}, steps {a.steps}, batch {B} + {BU} unlabeled, device {device}")
    if a.smoke:                       # visual check: rows = original, weak, strong (8 labeled crops)
        import torchvision.utils as vu
        xs = to_float(d["x"][d["lab"][:8]])
        vu.save_image(torch.cat([xs, weak(xs), strong(xs)]), run_dir / "aug_check.png", nrow=8)
        log(f"augmentation sample written to {run_dir / 'aug_check.png'}")
    t0, curve = time.time(), []
    model.train()
    for step in range(a.steps):
        li = d["lab"][torch.randint(0, len(d["lab"]), (B,), device=device)]
        xl, yl = to_float(d["x"][li]), d["y"][li]
        ramp = min(1.0, step / (0.4 * a.steps))
        stats = {}
        if a.method == "supervised":
            loss = F.cross_entropy(model(to_input(weak(xl))), yl)
        else:
            ui = torch.randint(0, n_u, (BU,), device=device)
            xu = to_float(d["x"][ui])
            if a.method in ("fixmatch", "flexmatch"):
                with torch.no_grad():                                          # pseudo-labels only
                    lw = model(to_input(weak(xu)))
                logits = model(to_input(torch.cat([weak(xl), strong(xu)])))
                ll, ls = logits[:B], logits[B:]
                probs = torch.softmax(lw, 1)
                conf, pseudo = probs.max(1)
                if a.method == "fixmatch":
                    mask = (conf >= tau).float()
                else:
                    cnt = torch.bincount(selected[selected >= 0], minlength=len(CLASSES)).float()
                    unused = (selected < 0).sum().float()
                    denom = torch.maximum(cnt.max(), unused).clamp_min(1.0)   # threshold warm-up
                    beta = cnt / denom
                    mask = (conf >= tau * (beta[pseudo] / (2.0 - beta[pseudo]))).float()
                    over = conf >= tau
                    selected[ui[over]] = pseudo[over]
                lu = (F.cross_entropy(ls, pseudo, reduction="none") * mask).mean()
                loss = F.cross_entropy(ll, yl) + lu
                stats = {"mask": float(mask.mean())}
            elif a.method == "pseudolabel":
                logits = model(to_input(torch.cat([weak(xl), weak(xu)])))
                ll, lu_ = logits[:B], logits[B:]
                conf, pseudo = torch.softmax(lu_.detach(), 1).max(1)
                mask = (conf >= tau).float()
                lu = (F.cross_entropy(lu_, pseudo, reduction="none") * mask).mean()
                loss = F.cross_entropy(ll, yl) + ramp * lu
                stats = {"mask": float(mask.mean())}
            else:                                                              # meanteacher
                with torch.no_grad():
                    ema.train()
                    target = torch.softmax(ema(to_input(weak(xu))), 1)
                logits = model(to_input(torch.cat([weak(xl), weak(xu)])))
                ll, lu_ = logits[:B], logits[B:]
                lu = F.mse_loss(torch.softmax(lu_, 1), target, reduction="none").mean(1).mean()
                loss = F.cross_entropy(ll, yl) + 50.0 * ramp * lu
        if not torch.isfinite(loss):
            log(f"FAILED non-finite loss at step {step}")
            raise SystemExit(1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        ema_update(ema, model, 0.999)
        if a.eval_every and (step + 1) % a.eval_every == 0:
            t1, f1 = evaluate(ema, d)
            curve.append({"step": step + 1, "top1": t1, "macro_f1": f1})
            extra = " ".join(f"{k} {v:.3f}" for k, v in stats.items())
            log(f"step {step + 1}/{a.steps} loss {loss.item():.4f} {extra} lr {sched.get_last_lr()[0]:.4f} "
                f"EMA test top1 {t1:.4f} macro F1 {f1:.4f} ({(time.time() - t0) / 60:.1f} min)")
    t1, f1 = evaluate(ema, d)
    res = {"tag": a.tag, "method": a.method, "ratio": a.ratio, "init": a.init, "steps": a.steps,
           "labeled_images": d["n_lab_imgs"], "labeled_crops": d["n_lab"], "ema_top1": t1, "ema_macro_f1": f1,
           "minutes": round((time.time() - t0) / 60, 1), "curve": curve}
    (run_dir / "result.json").write_text(json.dumps(res, indent=1))
    log(f"final EMA test top1 {t1:.4f} macro F1 {f1:.4f}")
    log(f"DONE {a.tag}")


if __name__ == "__main__":
    main()
