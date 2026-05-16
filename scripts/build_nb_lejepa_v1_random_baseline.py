"""
Builder for nb_lejepa_v1_random_encoder_baseline.ipynb.

LeJEPA v1 POC validation: random-encoder linear-probe baseline.

Why this notebook exists:
- LeJEPA v1 ran 100 epochs (May 2-3 2026) producing val_acc 0.2377 on
  Tiny-ImageNet linear probe.
- 23.77% is 47× chance (200 classes), but we don't know how much comes from
  "JEPA-trained encoder is informative" vs "any high-dim encoder + linear
  probe can memorize 200 classes from 100K images."
- This notebook runs the matched control: same architecture, RANDOMLY-initialized
  encoder (no training), train linear probe for same 100 epochs, compare.

Decision matrix:
  random val_acc < 0.05 (10× chance)  →  🟢 all 0.24 gain is JEPA-trained
  0.05 ≤ random < 0.15                 →  🟡 partial contribution from JEPA
  random ≥ 0.15                        →  🔴 most of 0.24 is "any-encoder"; JEPA POC inconclusive

Compute: ~1-2h on T4 (no encoder training, only linear probe; downloads Tiny-ImageNet).

HARD RULES applied:
- Drive checkpoint linear probe head every epoch (small ~800KB)
- Resume-safe
- Match LeJEPA v1 recipe: ViT-Small/patch8, img_size=128, num_classes=512
"""
import json
from pathlib import Path

NOTEBOOKS_DIR = Path("/Volumes/SSD Major/fish/openinterp-work/notebooks")


def code(lines, **meta):
    return {"cell_type": "code", "metadata": meta or {}, "execution_count": None,
            "outputs": [], "source": [l + "\n" for l in lines]}


def md(lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def build():
    cells = []

    cells.append(md([
        "# LeJEPA v1 — Random-Encoder Linear-Probe Baseline",
        "",
        "**Why**: LeJEPA v1 (run May 2-3 2026) reported val_acc 0.2377 on Tiny-ImageNet linear probe.",
        "This is 47× chance (200 classes), but we need the matched control to know how much",
        "comes from JEPA training vs how much any random encoder + linear probe achieves.",
        "",
        "**Decision matrix**:",
        "- random val_acc < 0.05 → 🟢 all 0.24 gain is JEPA-trained",
        "- 0.05 ≤ random < 0.15 → 🟡 partial contribution from JEPA",
        "- random ≥ 0.15 → 🔴 most of 0.24 is 'any-encoder'; POC inconclusive",
        "",
        "**Compute**: ~1-2h on T4. No encoder training; only linear probe head.",
    ]))

    cells.append(md(["## 1. Drive + paths"]))
    cells.append(code([
        "from pathlib import Path",
        "from google.colab import drive",
        "drive.mount('/content/drive', force_remount=False)",
        "",
        "DRIVE = Path('/content/drive/MyDrive')",
        "OUT = DRIVE / 'openinterp_runs' / 'lejepa_v1_random_baseline'",
        "OUT.mkdir(parents=True, exist_ok=True)",
        "RESULT_JSON = OUT / 'random_encoder_baseline_results.json'",
        "print(f'OUT: {OUT}')",
    ]))

    cells.append(md(["## 2. Install (timm + datasets)"]))
    cells.append(code([
        "import sys, subprocess",
        "def pip(*a): return subprocess.run([sys.executable, '-m', 'pip', *a], check=False)",
        "for mod in ('timm', 'datasets'):",
        "    try: __import__(mod)",
        "    except ImportError: pip('install', '-q', mod)",
        "",
        "import torch, timm",
        "print(f'torch {torch.__version__}, timm {timm.__version__}, cuda {torch.cuda.is_available()}')",
    ]))

    cells.append(md(["## 3. Config (matches LeJEPA v1 architecture)"]))
    cells.append(code([
        "IMG_SIZE     = 128",
        "PATCH_SIZE   = 8",
        "ENC_DIM      = 512   # ViT-Small num_classes output",
        "N_CLASSES    = 200   # Tiny-ImageNet",
        "BATCH_SIZE   = 256",
        "N_EPOCHS     = 100",
        "LR           = 2e-3",
        "WD           = 5e-2",
        "SEED         = 42",
        "",
        "torch.manual_seed(SEED)",
        "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "print(f'device={DEVICE}')",
    ]))

    cells.append(md(["## 4. Build RANDOMLY-INITIALIZED encoder (no training)"]))
    cells.append(code([
        "import timm",
        "import torch.nn as nn",
        "",
        "# Match LeJEPA v1 encoder exactly: ViT-Small/patch8, img_size=128, num_classes=ENC_DIM",
        "torch.manual_seed(SEED)",
        "encoder = timm.create_model(",
        "    'vit_small_patch8_224',  # closest to ViT-S/patch8 in timm",
        "    pretrained=False,         # CRITICAL: random init, no pretraining",
        "    img_size=IMG_SIZE,",
        "    num_classes=ENC_DIM,",
        ").to(DEVICE)",
        "",
        "# FREEZE encoder — random weights, never updated",
        "for p in encoder.parameters():",
        "    p.requires_grad = False",
        "encoder.eval()",
        "n_params = sum(p.numel() for p in encoder.parameters())",
        "print(f'Encoder: {n_params/1e6:.1f}M params, frozen, random init seed={SEED}')",
    ]))

    cells.append(md(["## 5. Tiny-ImageNet dataloaders"]))
    cells.append(code([
        "from datasets import load_dataset",
        "from torch.utils.data import DataLoader",
        "from torchvision import transforms",
        "",
        "ds = load_dataset('zh-plus/tiny-imagenet')",
        "print(f'train: {len(ds[\"train\"])}, valid: {len(ds[\"valid\"])}')",
        "",
        "train_tf = transforms.Compose([",
        "    transforms.Resize((IMG_SIZE, IMG_SIZE)),",
        "    transforms.RandomHorizontalFlip(),",
        "    transforms.ToTensor(),",
        "    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),",
        "])",
        "val_tf = transforms.Compose([",
        "    transforms.Resize((IMG_SIZE, IMG_SIZE)),",
        "    transforms.ToTensor(),",
        "    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),",
        "])",
        "",
        "def collate_factory(tf):",
        "    def collate(batch):",
        "        imgs = torch.stack([tf(b['image'].convert('RGB')) for b in batch])",
        "        labels = torch.tensor([b['label'] for b in batch], dtype=torch.long)",
        "        return imgs, labels",
        "    return collate",
        "",
        "train_loader = DataLoader(ds['train'], batch_size=BATCH_SIZE, shuffle=True,",
        "                          collate_fn=collate_factory(train_tf), num_workers=2)",
        "val_loader   = DataLoader(ds['valid'], batch_size=BATCH_SIZE, shuffle=False,",
        "                          collate_fn=collate_factory(val_tf), num_workers=2)",
        "print(f'batches: train={len(train_loader)}, val={len(val_loader)}')",
    ]))

    cells.append(md(["## 6. Linear probe head + training loop (matches LeJEPA recipe)"]))
    cells.append(code([
        "import torch.optim as optim",
        "from torch.optim.lr_scheduler import CosineAnnealingLR",
        "import json, time",
        "",
        "probe = nn.Linear(ENC_DIM, N_CLASSES).to(DEVICE)",
        "opt = optim.AdamW(probe.parameters(), lr=LR, weight_decay=WD)",
        "sched = CosineAnnealingLR(opt, T_max=N_EPOCHS)",
        "",
        "history = []",
        "",
        "@torch.no_grad()",
        "def eval_acc(probe, loader):",
        "    probe.eval()",
        "    correct = total = 0",
        "    for x, y in loader:",
        "        x, y = x.to(DEVICE), y.to(DEVICE)",
        "        feat = encoder(x).detach()",
        "        pred = probe(feat).argmax(-1)",
        "        correct += (pred == y).sum().item()",
        "        total += y.size(0)",
        "    return correct / total",
        "",
        "ckpt_path = OUT / 'probe_latest.pt'",
        "start_epoch = 0",
        "if ckpt_path.exists():",
        "    ck = torch.load(ckpt_path, map_location=DEVICE)",
        "    probe.load_state_dict(ck['probe'])",
        "    opt.load_state_dict(ck['opt'])",
        "    sched.load_state_dict(ck['sched'])",
        "    start_epoch = ck['epoch'] + 1",
        "    history = ck['history']",
        "    print(f'resumed from epoch {start_epoch}')",
        "",
        "t0 = time.time()",
        "for epoch in range(start_epoch, N_EPOCHS):",
        "    probe.train()",
        "    ep_loss = 0.0; nb = 0",
        "    for x, y in train_loader:",
        "        x, y = x.to(DEVICE), y.to(DEVICE)",
        "        with torch.no_grad():",
        "            feat = encoder(x)",
        "        logits = probe(feat)",
        "        loss = nn.functional.cross_entropy(logits, y)",
        "        opt.zero_grad(); loss.backward(); opt.step()",
        "        ep_loss += loss.item(); nb += 1",
        "    sched.step()",
        "    val_acc = eval_acc(probe, val_loader)",
        "    history.append({'epoch': epoch, 'probe_loss': ep_loss/nb, 'val_acc': val_acc,",
        "                    'lr': sched.get_last_lr()[0]})",
        "    print(f'epoch {epoch:3d}: loss={ep_loss/nb:.4f}  val_acc={val_acc:.4f}  ({time.time()-t0:.0f}s elapsed)')",
        "    # checkpoint every epoch",
        "    torch.save({'probe': probe.state_dict(), 'opt': opt.state_dict(),",
        "                'sched': sched.state_dict(), 'epoch': epoch, 'history': history},",
        "               ckpt_path)",
        "",
        "with open(RESULT_JSON, 'w') as f:",
        "    json.dump({'final_val_acc': history[-1]['val_acc'], 'history': history,",
        "               'config': {'seed': SEED, 'n_epochs': N_EPOCHS, 'lr': LR, 'wd': WD,",
        "                          'enc_dim': ENC_DIM, 'n_classes': N_CLASSES,",
        "                          'img_size': IMG_SIZE, 'batch_size': BATCH_SIZE,",
        "                          'encoder': 'vit_small_patch8_224 (timm), random init, frozen'}},",
        "              f, indent=2)",
        "print(f'\\n✓ saved {RESULT_JSON}')",
    ]))

    cells.append(md(["## 7. Verdict"]))
    cells.append(code([
        "import json",
        "rnd = json.load(open(RESULT_JSON))['final_val_acc']",
        "lejepa = 0.2377  # LeJEPA v1 final val_acc from history.json epoch 99",
        "chance = 1.0 / N_CLASSES",
        "",
        "print(f'Random-encoder linear-probe val_acc: {rnd:.4f}')",
        "print(f'LeJEPA-encoder linear-probe val_acc: {lejepa:.4f}')",
        "print(f'Chance:                              {chance:.4f}')",
        "print(f'')",
        "print(f'Random / chance:  {rnd/chance:.1f}×')",
        "print(f'LeJEPA / chance:  {lejepa/chance:.1f}×')",
        "print(f'LeJEPA / Random:  {lejepa/rnd:.1f}×  ← signal-of-JEPA-training ratio')",
        "print(f'')",
        "",
        "if rnd < 0.05:",
        "    print('🟢 VERDICT: random < 0.05 — all JEPA val_acc gain is from JEPA training. POC validated.')",
        "elif rnd < 0.15:",
        "    print(f'🟡 VERDICT: random in [0.05, 0.15] — partial JEPA contribution. LeJEPA/Random = {lejepa/rnd:.1f}×.')",
        "else:",
        "    print(f'🔴 VERDICT: random ≥ 0.15 — most of the 0.24 was \"any encoder\" effect. POC inconclusive.')",
    ]))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_path = NOTEBOOKS_DIR / "nb_lejepa_v1_random_encoder_baseline.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {out_path} ({len(cells)} cells, {out_path.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    build()
