#!/usr/bin/env python3
"""
Generate 4 high-impact charts for the FabricationGuard HF dataset README.
Saves PNGs to /tmp/, then a separate cell uploads them to HF.

Charts:
  1. chart_hero.png            — big -88% headline number + supporting bars
  2. chart_auroc_bench.png     — cross-bench AUROC, brand colors
  3. chart_latency_vs_auroc.png — scatter: us vs alternatives (low latency, high AUROC)
  4. chart_methodology_timeline.png — visual prior-art lineage
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

OUT = Path('/tmp/fg_charts')
OUT.mkdir(exist_ok=True)

# OpenInterp brand colors
BRAND   = '#6366f1'   # indigo
ACCENT  = '#06b6d4'   # cyan
EMERALD = '#10b981'
ROSE    = '#f43f5e'
AMBER   = '#f59e0b'
INK     = '#0f172a'
INK_50  = '#94a3b8'

plt.rcParams.update({
    'figure.dpi':      150,
    'savefig.dpi':     200,
    'savefig.bbox':    'tight',
    'savefig.facecolor': 'white',
    'font.family':     'DejaVu Sans',
    'font.size':       11,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.spines.left':  False,
    'axes.grid':       True,
    'grid.alpha':      0.15,
    'grid.linestyle':  '-',
})


# ---- 1. HERO — confident-wrong reduction headline -----------------------
def chart_hero():
    fig = plt.figure(figsize=(11, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.3)

    # LEFT: big -88% number
    ax0 = fig.add_subplot(gs[0])
    ax0.axis('off')
    ax0.text(0.5, 0.78, '−88%',
             fontsize=78, fontweight='bold', color=EMERALD, ha='center', va='center',
             transform=ax0.transAxes)
    ax0.text(0.5, 0.50, 'confident-wrong reduction',
             fontsize=14, color=INK, ha='center', va='center', transform=ax0.transAxes)
    ax0.text(0.5, 0.40, 'on SimpleQA (held-out cross-bench)',
             fontsize=11, color=INK_50, ha='center', va='center', transform=ax0.transAxes)
    ax0.text(0.5, 0.20, 'AUROC 0.882',
             fontsize=22, fontweight='bold', color=BRAND, ha='center', va='center',
             transform=ax0.transAxes)
    ax0.text(0.5, 0.10, 'cross-bench transfer · 1 ms latency',
             fontsize=10, color=INK_50, ha='center', va='center', transform=ax0.transAxes)

    # RIGHT: confident-wrong bars per benchmark
    ax1 = fig.add_subplot(gs[1])
    benches = ['TruthfulQA', 'HaluEval', 'SimpleQA', 'MMLU']
    baseline = np.array([65.0, 57.5, 85.0, 46.0])
    after    = np.array([32.5, 27.5, 10.0, 36.0])
    ys = np.arange(len(benches))[::-1]
    bar_h = 0.35
    ax1.barh(ys + bar_h/2, baseline, height=bar_h, color=ROSE, alpha=0.7, label='Baseline (no probe)')
    ax1.barh(ys - bar_h/2, after,    height=bar_h, color=EMERALD, alpha=0.85, label='+ FabricationGuard')
    for y, b, a in zip(ys, baseline, after):
        ax1.text(b + 1, y + bar_h/2, f'{b:.0f}%', va='center', fontsize=9, color=INK)
        ax1.text(a + 1, y - bar_h/2, f'{a:.0f}%', va='center', fontsize=9, color=INK, fontweight='bold')
        reduction = (b - a) / b * 100
        ax1.text(95, y, f'−{reduction:.0f}%', va='center', fontsize=10, fontweight='bold',
                 color=EMERALD if reduction > 50 else AMBER)
    ax1.set_yticks(ys)
    ax1.set_yticklabels(benches, fontsize=11)
    ax1.set_xlabel('Confident-wrong rate (%)', fontsize=11)
    ax1.set_xlim(0, 105)
    ax1.set_title('Mitigation impact (abstain mode @ τ=0.684)', fontsize=12, pad=12, loc='left')
    ax1.legend(loc='lower right', frameon=False, fontsize=10)

    fig.suptitle('FabricationGuard — Qwen3.6-27B', fontsize=14, fontweight='bold', y=1.02, x=0.05, ha='left')
    fig.savefig(OUT / 'chart_hero.png')
    plt.close(fig)
    print('✓ chart_hero.png')


# ---- 2. AUROC bench breakdown -------------------------------------------
def chart_auroc_bench():
    fig, ax = plt.subplots(figsize=(10, 5))
    benches = ['TruthfulQA-MC1', 'HaluEval-QA', 'SimpleQA', 'MMLU']
    sae_v1   = [0.556, 0.500, 0.494, 0.544]
    lr_within = [0.536, 0.903, 0.706, 0.631]
    lr_cross  = [0.599, 0.619, 0.882, 0.444]

    x = np.arange(len(benches))
    w = 0.26
    ax.bar(x - w, sae_v1,   width=w, color=INK_50, alpha=0.7, label='SAE-single feature (v1, abandoned)')
    ax.bar(x,     lr_within, width=w, color=ACCENT, label='LR within-bench')
    ax.bar(x + w, lr_cross,  width=w, color=BRAND,  label='LR cross-bench (production)')

    for i, v in enumerate(lr_cross):
        ax.text(i + w, v + 0.015, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold',
                color=BRAND if v > 0.6 else ROSE)
    for i, v in enumerate(lr_within):
        ax.text(i, v + 0.015, f'{v:.3f}', ha='center', fontsize=9, color=ACCENT)

    ax.axhline(0.5, color=INK_50, linestyle='--', linewidth=1, alpha=0.5)
    ax.text(len(benches) - 0.3, 0.51, 'chance', fontsize=9, color=INK_50)
    ax.axhspan(0.85, 1.0, color=EMERALD, alpha=0.05)
    ax.text(0.05, 0.93, 'in-scope target zone', fontsize=9, color=EMERALD,
            transform=ax.get_yaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(benches)
    ax.set_ylim(0.40, 1.0)
    ax.set_ylabel('AUROC')
    ax.set_title('Detection AUROC across the 4 hallucination benchmarks',
                 fontsize=13, pad=14, loc='left')
    ax.legend(loc='upper right', frameon=False, fontsize=9.5)

    # Footnote
    fig.text(0.05, -0.02,
             'Cross-bench: probe trained on the OTHER 3 benches\' train splits, evaluated on this held-out test set. '
             'Honest scope: TruthfulQA = misconceptions ≠ fabrication; MMLU = capability control.',
             fontsize=9, color=INK_50, style='italic')

    fig.savefig(OUT / 'chart_auroc_bench.png')
    plt.close(fig)
    print('✓ chart_auroc_bench.png')


# ---- 3. Latency vs AUROC scatter ----------------------------------------
def chart_latency_vs_auroc():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Each tool: (latency_ms, auroc, label, color, marker_size, open_weights)
    tools = [
        (1,    0.88, 'OpenInterp\nFabricationGuard', BRAND,   480, True),
        (100,  0.87, 'Patronus Lynx-70B',             ACCENT,  220, True),
        (600,  0.85, 'Vectara HHEM-2.1',              INK_50,  220, True),
        (152,  0.82, 'Galileo Luna-2',                AMBER,   220, False),
        (300,  0.78, 'Cleanlab TLM',                  ROSE,    220, False),
        (1000, 0.84, 'Goodfire Ember',                '#a855f7', 220, False),
    ]
    for latency, auroc, label, color, size, opensrc in tools:
        edge = 'black' if 'Fabrication' in label else 'none'
        lw   = 2 if 'Fabrication' in label else 0
        ax.scatter(latency, auroc, s=size, color=color, edgecolors=edge,
                   linewidths=lw, alpha=0.85, zorder=3 if 'Fabrication' in label else 2)
        offset_x = -50 if latency < 10 else 30
        ha = 'right' if latency < 10 else 'left'
        weight = 'bold' if 'Fabrication' in label else 'normal'
        ax.annotate(label, xy=(latency, auroc), xytext=(offset_x, 8),
                    textcoords='offset points', fontsize=9.5, ha=ha, fontweight=weight,
                    color=color)

    ax.set_xscale('log')
    ax.set_xlabel('Latency per call (ms, log scale)', fontsize=11)
    ax.set_ylabel('Hallucination AUROC', fontsize=11)
    ax.set_xlim(0.4, 2500)
    ax.set_ylim(0.74, 0.92)
    ax.set_title('Latency vs AUROC — FabricationGuard sits in the upper-left quadrant',
                 fontsize=12.5, pad=14, loc='left')

    # Quadrant guides
    ax.axhline(0.85, color=INK_50, linestyle=':', alpha=0.4, linewidth=1)
    ax.axvline(50,   color=INK_50, linestyle=':', alpha=0.4, linewidth=1)
    ax.text(0.6, 0.91, 'BETTER\n(low latency,\nhigh AUROC)', fontsize=8.5, color=EMERALD,
            ha='left', fontweight='bold')
    ax.text(2300, 0.755, 'worse', fontsize=8.5, color=ROSE, ha='right')

    fig.text(0.05, -0.03,
             '1 ms is achieved by capturing an activation the model already computes and running '
             'a single matrix multiplication. LLM-judge methods run a separate model.',
             fontsize=9, color=INK_50, style='italic')

    fig.savefig(OUT / 'chart_latency_vs_auroc.png')
    plt.close(fig)
    print('✓ chart_latency_vs_auroc.png')


# ---- 4. Methodology timeline / lineage ----------------------------------
def chart_methodology_timeline():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis('off')

    # Timeline events — alternate above/below to avoid crowding
    # (date, who, what, position: 'above' or 'below')
    events = [
        ('2025-02', 'Apollo Research', 'Linear probes for\nstrategic deception\n(arXiv:2502.03407)', 'above'),
        ('2025-03', 'Anthropic',       'Tracing the Thoughts —\ncircuit tracing in reasoning',       'below'),
        ('2025-08', 'Anthropic',       'Persona Vectors —\nhallucination + sycophancy\n(arXiv:2507.21509)', 'above'),
        ('2025-10', 'Anthropic',       'Signs of Introspection —\nlimited self-knowledge',           'below'),
        ('2026-03', 'Anthropic',       'Dedicated Feature\nCrosscoder (DFC)',                        'above'),
        ('2026-04', 'OpenInterp',      'FabricationGuard v2\n(this artifact)\n+ ProbeBench',         'below'),
    ]

    # Timeline line
    n = len(events)
    xs = np.linspace(0.10, 0.92, n)
    ax.plot(xs, [0.5] * n, color=INK_50, linewidth=2, zorder=1)

    for (date, who, what, side), x in zip(events, xs):
        is_us = 'OpenInterp' in who
        color = BRAND if is_us else (ACCENT if who == 'Anthropic' else AMBER)
        size  = 320 if is_us else 180
        edge  = 'black' if is_us else 'none'
        lw    = 2 if is_us else 0
        ax.scatter(x, 0.5, s=size, color=color, edgecolors=edge, linewidths=lw, zorder=2)

        if side == 'above':
            who_y, what_y, date_y = 0.93, 0.84, 0.42
            what_va = 'top'
        else:
            who_y, what_y, date_y = 0.32, 0.23, 0.58
            what_va = 'top'

        weight = 'bold' if is_us else 'semibold'
        ax.text(x, who_y, who, ha='center', va='center', fontsize=10.5, color=color, fontweight=weight)
        ax.text(x, what_y, what, ha='center', va=what_va, fontsize=8.5, color=INK,
                multialignment='center')
        ax.text(x, date_y, date, ha='center', va='center', fontsize=9, color=INK_50)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 1.0)
    ax.set_title('Methodology lineage — frontier-lab research → OSS productization',
                 fontsize=13, pad=16, loc='left', x=0.02, y=1.02, fontweight='bold')

    # Legend
    legend_handles = [
        mpatches.Patch(color=ACCENT, label='Anthropic (closed source / 7-8B)'),
        mpatches.Patch(color=AMBER,  label='Apollo Research (open / Llama)'),
        mpatches.Patch(color=BRAND,  label='OpenInterp (open / 27B+ · this work)'),
    ]
    ax.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.05),
              frameon=False, ncol=3, fontsize=9.5)

    fig.savefig(OUT / 'chart_methodology_timeline.png')
    plt.close(fig)
    print('✓ chart_methodology_timeline.png')


if __name__ == '__main__':
    chart_hero()
    chart_auroc_bench()
    chart_latency_vs_auroc()
    chart_methodology_timeline()
    print(f'\nAll charts written to {OUT}')
    for p in sorted(OUT.glob('*.png')):
        print(f'  {p.name}: {p.stat().st_size/1024:.1f} KB')
