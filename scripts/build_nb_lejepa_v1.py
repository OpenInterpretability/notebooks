"""
Builder for nb_lejepa_v1_tinyimagenet.ipynb.

JEPA POC v1 — LeJEPA puro em Tiny-ImageNet (HF dataset).

Reference: Balestriero & LeCun, "LeJEPA: Provable and Scalable Self-Supervised
Learning Without the Heuristics" (arxiv 2511.08544, Nov 2025).
Official repo: https://github.com/rbalestr-lab/lejepa

Architecture (faithful to official MINIMAL.md):
- Encoder: timm ViT-Small/patch8 (img_size=128, num_classes=512)
- Projection: MLP 512 → 2048 → 2048 → 16 (proj_dim)
- NO predictor, NO masking, NO teacher-student, NO EMA, NO stop-gradient
- 4 views per image (V=4)
- Loss: λ·SIGReg(proj) + (1-λ)·invariance_loss
- Linear probe joint with .detach() for monitoring

Hyperparameters (paper recipe):
- lambda=0.02, V=4, proj_dim=16
- AdamW lr=2e-3, weight_decay=5e-2
- bs=256 (auto-reduce to 64 on T4)
- bf16 mixed precision
- 100 epochs (paper uses 800; 100 sufficient for POC validation)
- Cosine schedule with linear warmup

Compute: ~4h A100 / ~12h T4 (with bs auto-adjust).
Output: caiovicentino1/lejepa-v1-tinyimagenet (HF model repo, optional).
Drive: /content/drive/MyDrive/openinterp_runs/lejepa_v1/

HARD RULES applied:
- Drive checkpoint every epoch (model + optimizer + epoch counter)
- Resume-safe: re-running training cell picks up from last epoch
- Auto-detect GPU (T4/L4/A100) and adjust batch_size accordingly
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
        "# LeJEPA v1 POC — Tiny-ImageNet (HF dataset)",
        "",
        "**Reference**: Balestriero & LeCun, [arxiv 2511.08544](https://arxiv.org/abs/2511.08544) (Nov 2025).",
        "",
        "**What this notebook does**:",
        "- Trains a self-supervised representation on Tiny-ImageNet (200 classes, 100K images)",
        "- Architecture: ViT-Small/patch8 + MLP projection (no predictor, no masking, no EMA)",
        "- Loss: `λ·SIGReg(proj) + (1-λ)·invariance_loss`",
        "- Joint linear probe (with `.detach()`) for monitoring downstream accuracy",
        "",
        "**Hyperparameters** (paper recipe):",
        "- λ=0.02, V=4 views, proj_dim=16",
        "- AdamW lr=2e-3, wd=5e-2, bf16",
        "- 100 epochs (paper uses 800; 100 enough for POC validation)",
        "- Batch auto-adjusts: 256 (A100) / 128 (L4) / 64 (T4)",
        "",
        "**Compute**: ~4h A100 / ~8h L4 / ~12h T4",
        "",
        "**HARD RULES** applied:",
        "- Drive checkpoint every epoch — `feedback_colab_must_checkpoint_or_dont_run.md`",
        "- Resume-safe: re-run training cell to continue from last epoch",
        "",
        "**Goal**: validate JEPA mechanics + SIGReg loss + establish linear-probe baseline before v2 cross-modal extension.",
    ]))

    # ============================================================
    # Cell 1 — Drive mount + paths
    # ============================================================
    cells.append(md(["## 1. Drive mount + resume paths"]))
    cells.append(code([
        "from pathlib import Path",
        "import os, json, time, math, shutil, random",
        "import torch, numpy as np",
        "",
        "from google.colab import drive",
        "drive.mount('/content/drive', force_remount=False)",
        "DRIVE = Path('/content/drive/MyDrive')",
        "OUT = DRIVE / 'openinterp_runs' / 'lejepa_v1'",
        "OUT.mkdir(parents=True, exist_ok=True)",
        "(OUT / 'ckpt').mkdir(parents=True, exist_ok=True)",
        "(OUT / 'logs').mkdir(parents=True, exist_ok=True)",
        "print(f'OUT: {OUT}')",
        "print(f'Existing: {sorted(p.name for p in OUT.iterdir())}')",
    ]))

    # ============================================================
    # Cell 2 — Install deps
    # ============================================================
    cells.append(md(["## 2. Install deps"]))
    cells.append(code([
        "# IMPORTANT: do NOT upgrade torch/torchvision on Colab — they come paired and",
        "# `pip install -U torch` breaks torchvision (mismatch causes 'operator torchvision::nms does not exist').",
        "!pip install -q timm datasets huggingface_hub matplotlib tqdm",
        "",
        "# lejepa not on PyPI — clone official repo and install from source with --no-deps",
        "# (--no-deps prevents re-installing torch via lejepa's transitive deps)",
        "import os, subprocess",
        "if not os.path.exists('/content/lejepa'):",
        "    subprocess.run(['git', 'clone', '--quiet', '--depth=1',",
        "                    'https://github.com/rbalestr-lab/lejepa.git', '/content/lejepa'], check=True)",
        "subprocess.run(['pip', 'install', '-q', '--no-deps', '-e', '/content/lejepa'], check=True)",
        "",
        "import timm, torch, torch.nn as nn, torch.nn.functional as F",
        "from torch.amp import autocast, GradScaler",
        "import lejepa",
        "print(f'torch {torch.__version__}, timm {timm.__version__}, lejepa loaded from source')",
    ]))

    # ============================================================
    # Cell 3 — GPU detect + CFG
    # ============================================================
    cells.append(md(["## 3. GPU auto-detect + CFG"]))
    cells.append(code([
        "# Defensive imports (safe to re-run cell out of order)",
        "import random, math, torch",
        "import numpy as np",
        "",
        "# GPU detection — adjust batch size per device",
        "gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'",
        "vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0",
        "print(f'GPU: {gpu_name}, VRAM: {vram_gb:.1f} GB')",
        "",
        "if 'A100' in gpu_name or vram_gb > 70:",
        "    BS = 256",
        "elif 'L4' in gpu_name or vram_gb > 20:",
        "    BS = 128",
        "elif 'T4' in gpu_name or vram_gb > 14:",
        "    BS = 64",
        "else:",
        "    BS = 32",
        "print(f'Auto batch size: {BS}')",
        "",
        "# ---- LeJEPA config (paper recipe) ----",
        "MODEL_NAME    = 'vit_small_patch8_224'    # backbone (img_size adjusted to 128)",
        "IMG_SIZE      = 128",
        "PROJ_DIM      = 16                        # MLP projection final dim",
        "BACKBONE_DIM  = 512                       # backbone num_classes",
        "MLP_HIDDEN    = 2048",
        "V             = 4                         # views per image",
        "LAMB          = 0.02                      # SIGReg weight (paper default)",
        "",
        "# ---- Training ----",
        "EPOCHS        = 100                       # paper uses 800; 100 sufficient for POC",
        "LR            = 2e-3",
        "WD            = 5e-2",
        "WARMUP_EPOCHS = 5",
        "GRAD_CLIP     = 1.0",
        "",
        "# ---- SIGReg ----",
        "NUM_SLICES    = 1024                      # random projections for SIGReg",
        "EPPS_T_MAX    = 3.0                       # Epps-Pulley integration range",
        "EPPS_N_POINTS = 17                        # must be odd",
        "",
        "# ---- Dataset ----",
        "DATASET_NAME  = 'Maysee/tiny-imagenet'    # 200 classes, 100K train / 10K val",
        "N_CLASSES     = 200",
        "",
        "# ---- Misc ----",
        "DEVICE        = 'cuda'",
        "DTYPE         = torch.bfloat16",
        "SEED          = 42",
        "torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)",
        "",
        "print(f'Config: V={V}, lambda={LAMB}, proj_dim={PROJ_DIM}, bs={BS}, epochs={EPOCHS}')",
    ]))

    # ============================================================
    # Cell 4 — Load Tiny-ImageNet + multi-view augmentation
    # ============================================================
    cells.append(md(["## 4. Tiny-ImageNet + multi-view augmentation (V=4 views per image)"]))
    cells.append(code([
        "from datasets import load_dataset",
        "from torchvision.transforms import v2",
        "from torch.utils.data import Dataset, DataLoader",
        "",
        "print(f'Loading {DATASET_NAME}...')",
        "ds = load_dataset(DATASET_NAME)",
        "print(f'Splits: {list(ds.keys())}')",
        "print(f'Train: {len(ds[\"train\"])}, Valid: {len(ds[\"valid\"])}')",
        "",
        "# LeJEPA augmentation pipeline (faithful to MINIMAL.md)",
        "AUG = v2.Compose([",
        "    v2.RandomResizedCrop(IMG_SIZE, scale=(0.08, 1.0)),",
        "    v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),",
        "    v2.RandomGrayscale(p=0.2),",
        "    v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),",
        "    v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),",
        "    v2.RandomHorizontalFlip(),",
        "    v2.ToImage(),",
        "    v2.ToDtype(torch.float32, scale=True),",
        "    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),",
        "])",
        "",
        "# Eval transform (deterministic, no aug)",
        "EVAL_TRANSFORM = v2.Compose([",
        "    v2.Resize(IMG_SIZE),",
        "    v2.CenterCrop(IMG_SIZE),",
        "    v2.ToImage(),",
        "    v2.ToDtype(torch.float32, scale=True),",
        "    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),",
        "])",
        "",
        "class MultiViewDataset(Dataset):",
        "    \"\"\"Returns V augmented views of each image plus the label.\"\"\"",
        "    def __init__(self, hf_dataset, transform, n_views=V):",
        "        self.ds = hf_dataset",
        "        self.tf = transform",
        "        self.V = n_views",
        "    def __len__(self):",
        "        return len(self.ds)",
        "    def __getitem__(self, idx):",
        "        item = self.ds[idx]",
        "        img = item['image'].convert('RGB')",
        "        views = torch.stack([self.tf(img) for _ in range(self.V)])  # (V, C, H, W)",
        "        return views, item['label']",
        "",
        "class EvalDataset(Dataset):",
        "    \"\"\"Single eval-transform view.\"\"\"",
        "    def __init__(self, hf_dataset, transform):",
        "        self.ds = hf_dataset",
        "        self.tf = transform",
        "    def __len__(self):",
        "        return len(self.ds)",
        "    def __getitem__(self, idx):",
        "        item = self.ds[idx]",
        "        return self.tf(item['image'].convert('RGB')), item['label']",
        "",
        "train_ds = MultiViewDataset(ds['train'], AUG, n_views=V)",
        "val_ds   = EvalDataset(ds['valid'], EVAL_TRANSFORM)",
        "",
        "train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)",
        "val_loader   = DataLoader(val_ds,   batch_size=BS*2, shuffle=False, num_workers=2, pin_memory=True)",
        "",
        "# Smoke test",
        "_views, _labels = next(iter(train_loader))",
        "print(f'Train batch: views={tuple(_views.shape)}, labels={tuple(_labels.shape)}')",
        "del _views, _labels",
    ]))

    # ============================================================
    # Cell 5 — Model: ViT encoder + MLP projection
    # ============================================================
    cells.append(md(["## 5. Model — ViT-Small/8 encoder + MLP projection"]))
    cells.append(code([
        "class MLP(nn.Module):",
        "    def __init__(self, in_dim, hidden_dims, norm_layer=nn.BatchNorm1d):",
        "        super().__init__()",
        "        layers = []",
        "        prev = in_dim",
        "        for h in hidden_dims[:-1]:",
        "            layers.extend([nn.Linear(prev, h), norm_layer(h), nn.GELU()])",
        "            prev = h",
        "        layers.append(nn.Linear(prev, hidden_dims[-1]))",
        "        self.net = nn.Sequential(*layers)",
        "    def forward(self, x):",
        "        return self.net(x)",
        "",
        "class LeJEPAModel(nn.Module):",
        "    def __init__(self, model_name=MODEL_NAME, img_size=IMG_SIZE,",
        "                 backbone_dim=BACKBONE_DIM, mlp_hidden=MLP_HIDDEN, proj_dim=PROJ_DIM):",
        "        super().__init__()",
        "        self.backbone = timm.create_model(",
        "            model_name, pretrained=False,",
        "            num_classes=backbone_dim, drop_path_rate=0.1, img_size=img_size,",
        "        )",
        "        self.proj = MLP(backbone_dim, [mlp_hidden, mlp_hidden, proj_dim])",
        "    def forward(self, x):",
        "        # x: (N, V, C, H, W)",
        "        N, V = x.shape[:2]",
        "        emb = self.backbone(x.flatten(0, 1))                # (N*V, backbone_dim)",
        "        proj = self.proj(emb).reshape(N, V, -1).transpose(0, 1)  # (V, N, proj_dim)",
        "        return emb, proj",
        "",
        "net = LeJEPAModel().to(DEVICE)",
        "n_params = sum(p.numel() for p in net.parameters()) / 1e6",
        "print(f'Model params: {n_params:.1f}M')",
        "print(f'VRAM after model init: {torch.cuda.memory_allocated()/1e9:.2f} GB')",
        "",
        "# Linear probe (joint training, monitors downstream acc)",
        "probe = nn.Sequential(nn.LayerNorm(BACKBONE_DIM), nn.Linear(BACKBONE_DIM, N_CLASSES)).to(DEVICE)",
        "print(f'Probe params: {sum(p.numel() for p in probe.parameters())/1e3:.1f}K')",
    ]))

    # ============================================================
    # Cell 6 — SIGReg loss (from lejepa lib)
    # ============================================================
    cells.append(md(["## 6. SIGReg loss — sketched isotropic Gaussian regularization"]))
    cells.append(code([
        "# SIGReg: project embeddings onto random 1D directions, test each against N(0,1)",
        "# via Epps-Pulley characteristic-function distance. Aggregate via mean.",
        "univariate_test = lejepa.univariate.EppsPulley(",
        "    t_max=EPPS_T_MAX, n_points=EPPS_N_POINTS, integration='trapezoid',",
        ").to(DEVICE)",
        "",
        "sigreg = lejepa.multivariate.SlicingUnivariateTest(",
        "    univariate_test=univariate_test,",
        "    num_slices=NUM_SLICES,",
        "    reduction='mean',",
        "    sampler='gaussian',",
        ").to(DEVICE)",
        "",
        "# Smoke test",
        "with torch.no_grad():",
        "    _x = torch.randn(256, PROJ_DIM, device=DEVICE)  # standard normal — should give low SIGReg",
        "    _stat_normal = sigreg(_x)",
        "    _y = torch.randn(256, PROJ_DIM, device=DEVICE) * 5 + 3  # shifted/scaled — should give high SIGReg",
        "    _stat_off = sigreg(_y)",
        "print(f'SIGReg(N(0,1)):       {_stat_normal.item():.4f} (should be small)')",
        "print(f'SIGReg(N(3,5)):       {_stat_off.item():.4f} (should be large)')",
        "assert _stat_off > _stat_normal, 'SIGReg sanity fail'",
        "print('✓ SIGReg detects deviation from N(0,1)')",
    ]))

    # ============================================================
    # Cell 7 — Optimizer + scheduler + scaler
    # ============================================================
    cells.append(md(["## 7. Optimizer + scheduler + scaler"]))
    cells.append(code([
        "from torch.optim.lr_scheduler import LambdaLR",
        "",
        "# All trainable params: backbone + proj + probe",
        "all_params = list(net.parameters()) + list(probe.parameters())",
        "opt = torch.optim.AdamW(all_params, lr=LR, weight_decay=WD, betas=(0.9, 0.999))",
        "",
        "steps_per_epoch = len(train_loader)",
        "total_steps = EPOCHS * steps_per_epoch",
        "warmup_steps = WARMUP_EPOCHS * steps_per_epoch",
        "",
        "def lr_lambda(step):",
        "    if step < warmup_steps:",
        "        return step / max(1, warmup_steps)",
        "    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)",
        "    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))",
        "",
        "scheduler = LambdaLR(opt, lr_lambda)",
        "scaler = GradScaler('cuda')",
        "",
        "print(f'Total steps: {total_steps:,} ({EPOCHS} epochs × {steps_per_epoch:,} steps)')",
    ]))

    # ============================================================
    # Cell 8 — Checkpoint utilities (Drive)
    # ============================================================
    cells.append(md(["## 8. Checkpoint utilities (Drive — HARD RULE)"]))
    cells.append(code([
        "import shutil",
        "from pathlib import Path",
        "import json, torch",
        "if 'OUT' not in dir():",
        "    OUT = Path('/content/drive/MyDrive/openinterp_runs/lejepa_v1')",
        "",
        "def _strip_compile_prefix(sd):",
        "    \"\"\"Remove _orig_mod. prefix from torch.compile() wrapped state_dict.\"\"\"",
        "    return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}",
        "",
        "def save_ckpt(epoch, train_loss, val_acc, sigreg_loss, inv_loss):",
        "    ckpt_path = OUT / 'ckpt' / f'lejepa_epoch_{epoch:03d}.pt'",
        "    latest_path = OUT / 'ckpt' / 'latest.pt'",
        "    torch.save({",
        "        'epoch': epoch,",
        "        'net_state': _strip_compile_prefix(net.state_dict()),",
        "        'probe_state': probe.state_dict(),",
        "        'opt_state': opt.state_dict(),",
        "        'scheduler_state': scheduler.state_dict(),",
        "        'scaler_state': scaler.state_dict(),",
        "        'train_loss': train_loss,",
        "        'val_acc': val_acc,",
        "        'sigreg_loss': sigreg_loss,",
        "        'inv_loss': inv_loss,",
        "    }, ckpt_path)",
        "    # also overwrite 'latest' for fast resume",
        "    shutil.copy(ckpt_path, latest_path)",
        "    print(f'  ✓ ckpt saved: {ckpt_path.name}')",
        "",
        "def load_latest():",
        "    \"\"\"Returns (start_epoch, history) — start_epoch=0 if no checkpoint.\"\"\"",
        "    latest = OUT / 'ckpt' / 'latest.pt'",
        "    history_path = OUT / 'logs' / 'history.json'",
        "    if not latest.exists():",
        "        return 0, []",
        "    state = torch.load(latest, map_location=DEVICE, weights_only=False)",
        "    # Strip _orig_mod. prefix in case ckpt was saved from torch.compile() wrapped model",
        "    sd = {k.replace('_orig_mod.', ''): v for k, v in state['net_state'].items()}",
        "    net.load_state_dict(sd)",
        "    probe.load_state_dict(state['probe_state'])",
        "    opt.load_state_dict(state['opt_state'])",
        "    scheduler.load_state_dict(state['scheduler_state'])",
        "    scaler.load_state_dict(state['scaler_state'])",
        "    history = json.loads(history_path.read_text()) if history_path.exists() else []",
        "    print(f'  ✓ resumed from epoch {state[\"epoch\"]}, val_acc={state[\"val_acc\"]:.4f}')",
        "    return state['epoch'] + 1, history",
        "",
        "print('Checkpoint utilities ready.')",
    ]))

    # ============================================================
    # Cell 9 — Eval function (linear probe on val set)
    # ============================================================
    cells.append(md(["## 9. Linear probe eval"]))
    cells.append(code([
        "@torch.no_grad()",
        "def eval_linear_probe():",
        "    net.eval(); probe.eval()",
        "    n_correct, n_total = 0, 0",
        "    for imgs, labels in val_loader:",
        "        imgs   = imgs.to(DEVICE, non_blocking=True)",
        "        labels = labels.to(DEVICE, non_blocking=True)",
        "        with autocast('cuda', dtype=DTYPE):",
        "            emb = net.backbone(imgs)",
        "            logits = probe(emb)",
        "        n_correct += (logits.argmax(dim=-1) == labels).sum().item()",
        "        n_total += labels.size(0)",
        "    net.train(); probe.train()",
        "    return n_correct / n_total",
    ]))

    # ============================================================
    # Cell 10 — Training loop (resume-safe, Drive ckpt every epoch)
    # ============================================================
    cells.append(md([
        "## 10. Training loop (resume-safe + Drive ckpt every epoch)",
        "",
        "Re-running this cell picks up from last checkpoint automatically.",
    ]))
    cells.append(code([
        "# Defensive imports — cell must work standalone after kernel restart",
        "import os, json, time, math, shutil",
        "from pathlib import Path",
        "import torch, torch.nn as nn, torch.nn.functional as F",
        "from torch.amp import autocast, GradScaler",
        "if 'OUT' not in dir():",
        "    OUT = Path('/content/drive/MyDrive/openinterp_runs/lejepa_v1')",
        "    print(f'(Re-established OUT={OUT})')",
        "",
        "from tqdm.auto import tqdm",
        "",
        "start_epoch, history = load_latest()",
        "print(f'Starting from epoch {start_epoch} → target {EPOCHS}')",
        "",
        "net.train(); probe.train()",
        "t0 = time.time()",
        "",
        "for epoch in range(start_epoch, EPOCHS):",
        "    epoch_losses = {'total': 0., 'sigreg': 0., 'inv': 0., 'probe': 0.}",
        "    n_batches = 0",
        "    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')",
        "",
        "    for vs, y in pbar:",
        "        vs = vs.to(DEVICE, non_blocking=True)  # (N, V, C, H, W)",
        "        y  = y.to(DEVICE, non_blocking=True)",
        "",
        "        with autocast('cuda', dtype=DTYPE):",
        "            emb, proj = net(vs)                # emb: (N*V, 512); proj: (V, N, proj_dim)",
        "            inv_loss = (proj.mean(0) - proj).square().mean()       # invariance: views close to mean",
        "            sigreg_loss = sigreg(proj.reshape(-1, PROJ_DIM))       # (V*N, proj_dim) → scalar",
        "            lejepa_loss = LAMB * sigreg_loss + (1 - LAMB) * inv_loss",
        "",
        "            # Joint linear probe (detached embedding)",
        "            y_rep = y.repeat_interleave(V)",
        "            probe_logits = probe(emb.detach())",
        "            probe_loss = F.cross_entropy(probe_logits, y_rep)",
        "",
        "            loss = lejepa_loss + probe_loss",
        "",
        "        opt.zero_grad(set_to_none=True)",
        "        scaler.scale(loss).backward()",
        "        scaler.unscale_(opt)",
        "        torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP)",
        "        scaler.step(opt)",
        "        scaler.update()",
        "        scheduler.step()",
        "",
        "        epoch_losses['total']  += loss.item()",
        "        epoch_losses['sigreg'] += sigreg_loss.item()",
        "        epoch_losses['inv']    += inv_loss.item()",
        "        epoch_losses['probe']  += probe_loss.item()",
        "        n_batches += 1",
        "        pbar.set_postfix(loss=f'{loss.item():.3f}', sig=f'{sigreg_loss.item():.3f}',",
        "                          inv=f'{inv_loss.item():.3f}', prb=f'{probe_loss.item():.3f}')",
        "",
        "    # Average epoch losses",
        "    for k in epoch_losses: epoch_losses[k] /= n_batches",
        "",
        "    # Eval linear probe",
        "    val_acc = eval_linear_probe()",
        "    elapsed = time.time() - t0",
        "    print(f'  Epoch {epoch+1}: loss={epoch_losses[\"total\"]:.4f} '",
        "          f'sigreg={epoch_losses[\"sigreg\"]:.4f} inv={epoch_losses[\"inv\"]:.4f} '",
        "          f'probe={epoch_losses[\"probe\"]:.4f} val_acc={val_acc:.4f} '",
        "          f'elapsed={elapsed/60:.1f}min')",
        "",
        "    # Save state",
        "    history.append({",
        "        'epoch': epoch,",
        "        **epoch_losses,",
        "        'val_acc': val_acc,",
        "        'lr': scheduler.get_last_lr()[0],",
        "    })",
        "    (OUT / 'logs' / 'history.json').write_text(json.dumps(history, indent=2))",
        "    save_ckpt(epoch, epoch_losses['total'], val_acc, epoch_losses['sigreg'], epoch_losses['inv'])",
        "",
        "print(f'\\nDone. Total time: {(time.time()-t0)/3600:.2f}h')",
    ]))

    # ============================================================
    # Cell 11 — Plots
    # ============================================================
    cells.append(md(["## 11. Plots — losses + linear probe accuracy"]))
    cells.append(code([
        "import matplotlib.pyplot as plt",
        "",
        "history = json.loads((OUT / 'logs' / 'history.json').read_text())",
        "epochs = [h['epoch']+1 for h in history]",
        "",
        "fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))",
        "",
        "ax = axes[0]",
        "ax.plot(epochs, [h['sigreg'] for h in history], label='SIGReg', color='#3b82f6')",
        "ax.plot(epochs, [h['inv'] for h in history],    label='Invariance', color='#10b981')",
        "ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_yscale('log')",
        "ax.set_title('LeJEPA component losses')",
        "ax.legend(); ax.grid(alpha=0.3)",
        "",
        "ax = axes[1]",
        "ax.plot(epochs, [h['probe'] for h in history], color='#f59e0b', label='Probe CE loss')",
        "ax.set_xlabel('Epoch'); ax.set_ylabel('Cross-entropy')",
        "ax.set_title('Linear probe loss (downstream signal)')",
        "ax.legend(); ax.grid(alpha=0.3)",
        "",
        "ax = axes[2]",
        "ax.plot(epochs, [h['val_acc']*100 for h in history], color='#ef4444', linewidth=2)",
        "ax.set_xlabel('Epoch'); ax.set_ylabel('Val acc (%)')",
        "ax.set_title('Linear probe val accuracy on Tiny-ImageNet')",
        "ax.grid(alpha=0.3)",
        "",
        "plt.tight_layout()",
        "fig_path = OUT / 'lejepa_v1_curves.png'",
        "plt.savefig(fig_path, dpi=170, bbox_inches='tight')",
        "plt.show()",
        "print(f'Saved: {fig_path}')",
    ]))

    # ============================================================
    # Cell 12 — Final verdict
    # ============================================================
    cells.append(md(["## 12. Final verdict"]))
    cells.append(code([
        "history = json.loads((OUT / 'logs' / 'history.json').read_text())",
        "best = max(history, key=lambda h: h['val_acc'])",
        "final = history[-1]",
        "",
        "print(f'=== LeJEPA v1 POC verdict ===')",
        "print(f'Final epoch: {final[\"epoch\"]+1}')",
        "print(f'Best val_acc:  {best[\"val_acc\"]:.4f} (epoch {best[\"epoch\"]+1})')",
        "print(f'Final val_acc: {final[\"val_acc\"]:.4f}')",
        "print(f'Final SIGReg loss: {final[\"sigreg\"]:.4f}')",
        "print(f'Final invariance loss: {final[\"inv\"]:.4f}')",
        "print()",
        "print('Reference baselines on Tiny-ImageNet (200 classes):')",
        "print('  Random:           0.5%')",
        "print('  ImageNet-pretrained ViT-S linear probe: ~50-55%')",
        "print('  SimCLR ViT-S from scratch (200 epochs): ~30-40%')",
        "print('  LeJEPA ViT-S/8 expected (~100 epochs):  ~25-35%')",
        "print()",
        "if best['val_acc'] >= 0.30:",
        "    print('🟢 STRONG — LeJEPA validated. Ready for v2 cross-modal extension.')",
        "elif best['val_acc'] >= 0.20:",
        "    print('🟡 MARGINAL — works but below SimCLR. Consider longer training.')",
        "else:",
        "    print('🔴 INSUFFICIENT — debug SIGReg balance or training setup.')",
    ]))

    # ============================================================
    # Save notebook
    # ============================================================
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path = NOTEBOOKS_DIR / "nb_lejepa_v1_tinyimagenet.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"✓ Wrote {out_path}")
    print(f"  Cells: {len(cells)}")


if __name__ == "__main__":
    build()
