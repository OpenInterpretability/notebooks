"""
PGAC Phase 2 — synthetic eval v2 with methodological fixes.

v1 failed because:
- BCE on 16K classes with 0.4% positive rate → probe predicts all-zero
- 67M params, 30K samples, 3 epochs = undertrained
- Treated as independent binary classification, not top-k retrieval

v2 fixes:
- SAE encoder warm-start (probe initialized to SAE projection, not random)
- Ranking-aware loss (margin between top-k and bottom)
- Smaller dictionary (d_sae=2048) for sanity check first
- Same-layer prediction sanity check (probe == SAE → AUROC = 1.0)
- Then cross-layer test (residual at L_n → SAE features at L_m)

Decision tree:
- Same-layer probe < 0.95 AUROC → wrong loss / training, debug
- Same-layer 0.95+, cross-layer < 0.6 → information not in upstream residual, abandon
- Same-layer 0.95+, cross-layer 0.85+ → methodology validates, proceed
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path('/Volumes/SSD Major/fish/openinterp-work/applications/pgac')
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# Smaller dictionary first — validate methodology before scaling
# ============================================================================
d_model = 1024          # smaller for synthetic, real Qwen is 4096
d_sae = 2048            # 2x expansion
k = 32                  # top-k active per token
n_train = 50000
n_test = 5000
n_epochs = 8

print('=== PGAC synthetic eval v2: methodology validation ===')
print(f'  d_model={d_model}, d_sae={d_sae}, k={k}')
print(f'  n_train={n_train}, n_test={n_test}')
print(f'  Sparsity: {k/d_sae:.4f} ({k}/{d_sae})')

# ============================================================================
# Synthetic data: known SAE encoder + cross-layer perturbation
# ============================================================================
# True SAE encoder: residual at layer L → features at layer L
W_sae = torch.randn(d_sae, d_model) * (1.0 / np.sqrt(d_model))  # unit-scale init

# Simulate "many layers between" by adding gradient-like perturbation
# Cross-layer: residual at L_n → features at L_m where m > n
# In real Qwen3.6, this is layer 11 → 31 = 20 layers of nonlinear computation
# Approximate as linear + noise (best-case for probe)
def cross_layer_perturbation(residuals, gap_layers=20, noise_std=0.5):
    """Simulate residual transformation through `gap_layers` of compute."""
    # Each layer adds small Gaussian-like update (residual stream pattern)
    h = residuals.clone()
    for _ in range(gap_layers):
        # Small linear update + nonlinearity + residual connection
        update = torch.randn(d_model, d_model) * 0.01
        h = h + 0.1 * F.gelu(h @ update)
    # Add noise to break perfect predictability
    h = h + torch.randn_like(h) * noise_std
    return h

# Sample residuals
residuals_train_low = torch.randn(n_train, d_model)  # at "L_n"
residuals_test_low = torch.randn(n_test, d_model)

# Compute features at L_n (same layer)
def compute_features(residuals, W, k_active, noise=0.0):
    logits = residuals @ W.T
    if noise > 0:
        logits = logits + torch.randn_like(logits) * noise
    _, topk_idx = torch.topk(logits, k=k_active, dim=-1)
    feature_active = torch.zeros_like(logits, dtype=torch.bool)
    feature_active.scatter_(-1, topk_idx, True)
    return feature_active.float(), logits

print('\n=== Same-layer prediction (probe = SAE encoder, sanity check) ===')

features_train_same, _ = compute_features(residuals_train_low, W_sae, k)
features_test_same, _ = compute_features(residuals_test_low, W_sae, k)

# Probe initialized to W_sae — should immediately give perfect prediction
class WarmStartProbe(nn.Module):
    def __init__(self, init_W):
        super().__init__()
        self.W = nn.Parameter(init_W.clone())
        self.b = nn.Parameter(torch.zeros(init_W.shape[0]))

    def forward(self, x):
        return x @ self.W.T + self.b

probe_same = WarmStartProbe(W_sae)
probe_same.eval()
with torch.no_grad():
    test_logits_same = probe_same(residuals_test_low)
    _, predicted_topk = torch.topk(test_logits_same, k=k, dim=-1)
    predicted_active = torch.zeros_like(test_logits_same, dtype=torch.bool)
    predicted_active.scatter_(-1, predicted_topk, True)
    intersection = (predicted_active & features_test_same.bool()).sum(dim=-1).float()
    recall_same = intersection / k
    print(f'  Same-layer recall@k (no training, perfect init): {recall_same.mean():.4f}')
# If this isn't ~1.0, our setup is broken
assert recall_same.mean() > 0.99, f'Sanity fail: same-layer recall = {recall_same.mean():.4f}'
print(f'  ✓ Sanity check passed — methodology can in principle work')

# ============================================================================
# Cross-layer prediction (the actually-interesting case)
# ============================================================================
print('\n=== Cross-layer prediction (residual at L_n → features at L_m) ===')

# Apply cross-layer perturbation to create "L_m" residual
residuals_train_high = cross_layer_perturbation(residuals_train_low, gap_layers=20, noise_std=0.3)
residuals_test_high = cross_layer_perturbation(residuals_test_low, gap_layers=20, noise_std=0.3)

# Features at L_m (using a slightly different SAE encoder for L_m)
W_sae_m = W_sae + torch.randn_like(W_sae) * 0.05  # ~5% drift in SAE structure
features_train_cross, _ = compute_features(residuals_train_high, W_sae_m, k, noise=0.1)
features_test_cross, _ = compute_features(residuals_test_high, W_sae_m, k, noise=0.1)

# Probe: predict features at L_m from residual at L_n
# Initialize with W_sae (best guess based on L_n's own SAE)
probe_cross = WarmStartProbe(W_sae_m)  # initialized with the true L_m projection (warm start)
optimizer = torch.optim.AdamW(probe_cross.parameters(), lr=1e-3, weight_decay=1e-5)

# Loss: ranking loss — top-k features must rank above bottom features
def topk_ranking_loss(logits, target_active, k):
    """Encourage top-k logits at active features."""
    pos_mask = target_active.bool()
    # Margin loss: top-k features should have logits > all others by margin 1.0
    # Approximate via pairwise: positive logit > average negative logit
    pos_logits = (logits * pos_mask.float()).sum(dim=-1) / k
    neg_logits = (logits * (~pos_mask).float()).sum(dim=-1) / (logits.shape[-1] - k)
    margin = pos_logits - neg_logits
    return F.softplus(-margin).mean()

print('  Training probe on residual_low → feature_high task...')
batch_size = 128
for epoch in range(n_epochs):
    indices = torch.randperm(n_train)
    losses = []
    for i in range(0, n_train, batch_size):
        batch_idx = indices[i:i+batch_size]
        x = residuals_train_low[batch_idx]
        y = features_train_cross[batch_idx]
        logits = probe_cross(x)
        loss = topk_ranking_loss(logits, y, k)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if epoch % 2 == 0 or epoch == n_epochs - 1:
        # Quick eval
        probe_cross.eval()
        with torch.no_grad():
            test_logits = probe_cross(residuals_test_low)
            _, predicted_topk = torch.topk(test_logits, k=k, dim=-1)
            predicted_active = torch.zeros_like(test_logits, dtype=torch.bool)
            predicted_active.scatter_(-1, predicted_topk, True)
            intersection = (predicted_active & features_test_cross.bool()).sum(dim=-1).float()
            recall = intersection / k
        probe_cross.train()
        print(f'  Epoch {epoch+1}/{n_epochs} loss: {np.mean(losses):.4f}  recall@k: {recall.mean():.4f}')

# Final eval
probe_cross.eval()
with torch.no_grad():
    test_logits = probe_cross(residuals_test_low)
    _, predicted_topk = torch.topk(test_logits, k=k, dim=-1)
    predicted_active = torch.zeros_like(test_logits, dtype=torch.bool)
    predicted_active.scatter_(-1, predicted_topk, True)

intersection = (predicted_active & features_test_cross.bool()).sum(dim=-1).float()
recall = intersection / k
print(f'\n  Cross-layer recall@k mean: {recall.mean():.4f}')
print(f'  recall@k > 0.7: {(recall > 0.7).float().mean():.2%}')
print(f'  recall@k > 0.85: {(recall > 0.85).float().mean():.2%}')

# Per-feature AUROC
test_logits_np = test_logits.detach().numpy()
features_test_np = features_test_cross.numpy()
aurocs = []
for f in np.random.choice(d_sae, size=200, replace=False):
    if features_test_np[:, f].sum() < 5:
        continue
    auc = roc_auc_score(features_test_np[:, f], test_logits_np[:, f])
    aurocs.append(auc)
aurocs = np.array(aurocs)
print(f'  Per-feature AUROC mean: {aurocs.mean():.4f}')
print(f'  AUROC > 0.85: {(aurocs > 0.85).mean():.2%}')

# ============================================================================
# Verdict
# ============================================================================
mean_recall = recall.mean().item()
mean_auroc = aurocs.mean()

print(f'\n=== Synthetic v2 verdict ===')
if mean_recall >= 0.85 and mean_auroc >= 0.85:
    verdict = '🟢 STRONG — ranking-loss + warm-start methodology validates. Real Colab run worth attempting.'
elif mean_recall >= 0.7 and mean_auroc >= 0.75:
    verdict = '🟡 MARGINAL — works in principle but tight margins. Real-world will likely be worse.'
elif mean_recall >= 0.4:
    verdict = '🟡 PARTIAL — significant cross-layer information loss; probe captures some signal but not enough'
else:
    verdict = '🔴 INSUFFICIENT — even with warm-start + ranking loss, cross-layer prediction fails on synthetic. Architecture revision needed.'

print(f'  {verdict}')
print(f'\n  recall@k = {mean_recall:.4f}')
print(f'  AUROC = {mean_auroc:.4f}')
print(f'  Required for PGAC quality preservation: recall ≥ 0.85, AUROC ≥ 0.85')

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))

ax = axes[0]
ax.hist(recall.numpy(), bins=30, color='#10b981', alpha=0.85, edgecolor='black')
ax.axvline(0.85, color='red', linestyle='--', label='Quality threshold')
ax.axvline(mean_recall, color='black', label=f'Mean = {mean_recall:.3f}')
ax.set_xlabel('Recall@k per token')
ax.set_ylabel('Count')
ax.set_title(f'Cross-layer recall (synthetic v2)\n d_sae={d_sae}, k={k}, gap=20 layers + noise')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
ax.hist(aurocs, bins=30, color='#3b82f6', alpha=0.85, edgecolor='black')
ax.axvline(0.85, color='red', linestyle='--', label='Quality threshold')
ax.axvline(mean_auroc, color='black', label=f'Mean = {mean_auroc:.3f}')
ax.set_xlabel('Per-feature AUROC')
ax.set_ylabel('Count')
ax.set_title('Per-feature AUROC (200 random features)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / 'pgac_synthetic_eval_v2.png', dpi=170, bbox_inches='tight')
print(f'\n✓ Saved: {OUT / "pgac_synthetic_eval_v2.png"}')
