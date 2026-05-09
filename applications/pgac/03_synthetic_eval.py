"""
PGAC Phase 2 — synthetic eval to validate methodology before real Colab run.

Tests: can a linear probe in principle predict top-k SAE features from a
residual stream? Uses synthetic data with known structure to validate the
training pipeline + measure achievable AUROC under ideal conditions.

If synthetic AUROC < 0.85 → methodology has fundamental issue, abort.
If synthetic AUROC ≥ 0.85 → real run is worth attempting.

Runtime: ~30 seconds on CPU.
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
# Realistic Qwen3.6-27B + SAE parameters
# ============================================================================
d_model = 4096          # Qwen3.6-27B residual dim
d_sae = 16384           # SAE expansion (using 4x for synthetic, real is 16x)
k = 64                  # TopK SAE k
n_train = 30000         # ~50 prompts × ~1024 tokens (realistic Phase 2 sample)
n_test = 5000

# ============================================================================
# Synthetic data generation — known ground truth
# ============================================================================
print('=== Generating synthetic residual + features (known structure) ===')

# True relationship: residuals are projected through random matrix to features
# This simulates the "L11 residual → L31 SAE features" mapping with some noise
# In reality, this matrix is a deep nonlinear function (layers 11-31 of Qwen3.6),
# but we approximate it as linear + noise for the probe-feasibility test

# True projection: residual → feature logits
W_true = torch.randn(d_sae, d_model) * 0.05  # encoder-like projection

# Sample residual streams (approximately Gaussian for real LLM residuals)
def sample_residuals(n, d):
    # Real LLM residuals have heavy-tailed distribution per Templeton 2024
    # Approximate with mixture of Gaussians + outliers
    base = torch.randn(n, d)
    outliers_mask = torch.rand(n, d) < 0.01
    base = base + outliers_mask * torch.randn(n, d) * 5
    return base

residuals_train = sample_residuals(n_train, d_model)
residuals_test = sample_residuals(n_test, d_model)

# Compute "true" feature logits (then apply TopK to get sparse activations)
def compute_topk_features(residuals, W, k_active):
    """Mimic SAE encoder: residuals → features → TopK."""
    # Linear projection
    logits = residuals @ W.T  # (n, d_sae)
    # Add noise to simulate the L11→L31 distance (more layers between, more noise)
    logits = logits + torch.randn_like(logits) * 0.5
    # TopK: keep only top-k absolute activations per row
    topk_vals, topk_indices = torch.topk(logits.abs(), k=k_active, dim=-1)
    # Build sparse representation
    feature_active = torch.zeros_like(logits, dtype=torch.bool)
    feature_active.scatter_(-1, topk_indices, True)
    return feature_active.float(), topk_indices

print(f'  Generating {n_train} train + {n_test} test samples...')
features_train, topk_train = compute_topk_features(residuals_train, W_true, k)
features_test, topk_test = compute_topk_features(residuals_test, W_true, k)

# Sanity: actual sparsity
print(f'  Sparsity (mean active per token): {features_train.mean():.4f} (expected: {k/d_sae:.4f})')

# ============================================================================
# Probe model: Linear classifier per feature (multi-label)
# Equivalent to nn.Linear(d_model, d_sae) trained with BCEWithLogitsLoss
# ============================================================================
class FeaturePresenceProbe(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W = nn.Linear(d_in, d_out, bias=True)

    def forward(self, x):
        return self.W(x)

probe = FeaturePresenceProbe(d_model, d_sae)

# Training
optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
batch_size = 128
n_epochs = 3

print(f'\n=== Training probe ({sum(p.numel() for p in probe.parameters())/1e6:.1f}M params) ===')
for epoch in range(n_epochs):
    indices = torch.randperm(n_train)
    losses = []
    for i in range(0, n_train, batch_size):
        batch_idx = indices[i:i+batch_size]
        x = residuals_train[batch_idx]
        y = features_train[batch_idx]  # (B, d_sae) binary
        logits = probe(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f'  Epoch {epoch+1}/{n_epochs} loss: {np.mean(losses):.4f}')

# ============================================================================
# Eval — recall@k, AUROC per feature, top-k overlap
# ============================================================================
print(f'\n=== Eval on test set ===')

probe.eval()
with torch.no_grad():
    test_logits = probe(residuals_test)  # (n_test, d_sae)

# Predicted top-k from probe
_, predicted_topk = torch.topk(test_logits, k=k, dim=-1)
predicted_active = torch.zeros_like(test_logits, dtype=torch.bool)
predicted_active.scatter_(-1, predicted_topk, True)

# True top-k
true_active = features_test.bool()

# Recall@k = fraction of true top-k features that the probe correctly predicts
intersection = (predicted_active & true_active).sum(dim=-1).float()  # (n,)
recall_at_k = intersection / k
print(f'  Recall@k mean: {recall_at_k.mean():.4f} (std {recall_at_k.std():.4f})')
print(f'  Recall@k > 0.7: {(recall_at_k > 0.7).float().mean():.2%}')
print(f'  Recall@k > 0.8: {(recall_at_k > 0.8).float().mean():.2%}')
print(f'  Recall@k > 0.9: {(recall_at_k > 0.9).float().mean():.2%}')

# AUROC per feature — sample 200 random features
print(f'\n  Computing AUROC per feature (200 random samples)...')
test_logits_np = test_logits.numpy()
true_active_np = features_test.numpy()

aurocs = []
for f in np.random.choice(d_sae, size=200, replace=False):
    if true_active_np[:, f].sum() < 5:
        continue  # skip features that fire too rarely for stable AUROC
    auc = roc_auc_score(true_active_np[:, f], test_logits_np[:, f])
    aurocs.append(auc)

aurocs = np.array(aurocs)
print(f'  Mean AUROC: {aurocs.mean():.4f} (median {np.median(aurocs):.4f})')
print(f'  AUROC > 0.85: {(aurocs > 0.85).mean():.2%}')
print(f'  AUROC > 0.95: {(aurocs > 0.95).mean():.2%}')

# ============================================================================
# Verdict
# ============================================================================
print(f'\n=== Synthetic methodology verdict ===')
mean_recall = recall_at_k.mean().item()
mean_auroc = aurocs.mean()

if mean_recall > 0.85 and mean_auroc > 0.85:
    verdict = '🟢 STRONG — methodology validates in synthetic ideal case. Real data run worth attempting.'
elif mean_recall > 0.7 and mean_auroc > 0.8:
    verdict = '🟡 ACCEPTABLE — methodology works but margins are tight. Real data needs to outperform synthetic.'
else:
    verdict = '🔴 INSUFFICIENT — methodology fails even in ideal synthetic case. Architecture or training needs revision.'

print(f'  {verdict}')
print(f'\n  → recall@k = {mean_recall:.4f}, mean AUROC = {mean_auroc:.4f}')
print(f'  → Required for PGAC quality preservation: recall ≥ 0.95, AUROC ≥ 0.95')

# ============================================================================
# Plot
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

ax = axes[0]
ax.hist(recall_at_k.numpy(), bins=30, color='#10b981', alpha=0.85, edgecolor='black')
ax.axvline(0.7, color='red', linestyle='--', label='Min for 8pp loss')
ax.axvline(mean_recall, color='black', label=f'Mean = {mean_recall:.2f}')
ax.set_xlabel('Recall@k per token')
ax.set_ylabel('Count')
ax.set_title('Probe recall@k distribution')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
ax.hist(aurocs, bins=30, color='#3b82f6', alpha=0.85, edgecolor='black')
ax.axvline(0.85, color='red', linestyle='--', label='Quality threshold')
ax.axvline(mean_auroc, color='black', label=f'Mean = {mean_auroc:.3f}')
ax.set_xlabel('Per-feature AUROC')
ax.set_ylabel('Count')
ax.set_title('AUROC distribution (200 random features)')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[2]
# Overlap heatmap: predicted top-k vs true top-k for first 100 test samples
overlap = (predicted_active[:100] & true_active[:100]).sum(dim=-1).numpy()
ax.scatter(np.arange(100), overlap, alpha=0.6, color='#10b981')
ax.axhline(k, color='red', linestyle='--', label=f'Perfect = {k}')
ax.axhline(0.7 * k, color='orange', linestyle=':', label='Min acceptable (70%)')
ax.set_xlabel('Test sample index')
ax.set_ylabel('Correctly predicted top-k features')
ax.set_title('Per-token top-k overlap')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / 'pgac_synthetic_eval.png', dpi=170, bbox_inches='tight')
print(f'\n✓ Saved: {OUT / "pgac_synthetic_eval.png"}')
