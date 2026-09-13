"""
Ablation chart for the manuscript (Figure: ablation_chart.png).

Why this file exists
--------------------
The submitted version of `ablation_chart.png` had no producing script anywhere
in the DISACT-GNN tree or in this repository -- it existed only as a PNG in
`lstm_new_features/sn-article-template/figures/`. It was rebuilt here so the
figure is reproducible from `results/ablations_5c.json`.

Two changes from the submitted figure, both forced by the five-country numbers:

1. The submitted chart starred "Events Only" as best. On the five-country
   sample the seven configurations that retain past-disappearance history span
   0.8080 to 0.8140 -- a range of 0.0060, inside the +/-0.003 test-set sampling
   noise measured by the paired bootstrap (exp7/exp10). Presenting that as a
   ranking is not supportable, so the star is gone and a zoom panel shows the
   overlap directly.

2. The submitted chart drew a dashed reference line at the Full Model value,
   which is itself one of the plotted bars, and drew the no-skill rate as a
   tenth bar. Bars are models; the no-skill rate is not. Both are removed.

Colour encodes the mechanism rather than the feature family: every
configuration that keeps past-disappearance counts performs the same, and the
two that drop them collapse.

Input:  results/ablations_5c.json  (9 feature sets x 3 seeds, Attention-LSTM)
Output: src/figures/ablation_chart.{png,pdf}
"""

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RESULTS_PATH = HERE.parent / 'results' / 'ablations_5c.json'
OUTPUT_DIR = HERE / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

# Short axis labels for the config strings in ablations_5c.json
LABELS = {
    'Events Only (No Network)':               'Events only',
    'Events + Clustering Features':           'Events + clustering',
    'Events + Structure Features':            'Events + structure',
    'Events + Centrality Features':           'Events + centrality',
    'Events + Weight Features':               'Events + edge weights',
    'Full Model (All Features)':              'Full model',
    'No Fatalities':                          'No fatalities',
    'No Past Disappearances (Early Warning)': 'No past disappearances',
    'Network Only (No Events)':               'Network only',
}

# Configurations that drop past-disappearance counts from the input
DROPS_HISTORY = {
    'No Past Disappearances (Early Warning)',
    'Network Only (No Events)',
}

FOCAL = 'Events Only (No Network)'

C_FOCAL = '#1f4e9c'    # events only
C_KEEPS = '#7fa6d9'    # keeps past-disappearance history
C_DROPS = '#e07b39'    # history removed
C_GREY = '#5a5a5a'

BASE, ANNOT, TICK = 9, 8, 7


def _style():
    mpl.rcParams.update({
        'font.size': BASE,
        'axes.titlesize': BASE,
        'axes.labelsize': BASE,
        'xtick.labelsize': TICK,
        'ytick.labelsize': TICK,
        'legend.fontsize': ANNOT,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })


def load_ablations(results_path=RESULTS_PATH):
    """Return ablation rows sorted by test AUPRC, descending."""
    with open(results_path) as f:
        payload = json.load(f)
    rows = sorted(payload['results'], key=lambda r: -r['test_auprc_mean'])
    return rows, payload['n_seeds']


def load_permutation_null_sd(path=None):
    """SD of the paired permutation null on the AUPRC difference (exp10)."""
    path = path or (HERE.parent / 'results' / 'exp10_bootstrap_events_only.json')
    with open(path) as f:
        return json.load(f)['permutation']['null_sd']


def load_random_baseline(path=None):
    """Prevalence-equivalent random-classifier AUPRC (test positive rate, exp10)."""
    path = path or (HERE.parent / 'results' / 'exp10_bootstrap_events_only.json')
    with open(path) as f:
        return json.load(f)['test_positive_rate']


def create_ablation_chart(results_path=RESULTS_PATH):
    rows, n_seeds = load_ablations(results_path)
    _style()

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(9.2, 4.2), gridspec_kw={'width_ratios': [1.55, 1.0]}
    )

    # ---------------- Panel a: all configurations ----------------
    y = range(len(rows))
    labels, means, sds, colors = [], [], [], []
    for r in rows:
        labels.append(LABELS[r['config']])
        means.append(r['test_auprc_mean'])
        sds.append(r['test_auprc_std'])
        if r['config'] in DROPS_HISTORY:
            colors.append(C_DROPS)
        elif r['config'] == FOCAL:
            colors.append(C_FOCAL)
        else:
            colors.append(C_KEEPS)

    ax_a.barh(list(y), means, xerr=sds, color=colors, height=0.72,
              error_kw=dict(ecolor=C_GREY, lw=0.9, capsize=2))
    ax_a.set_yticks(list(y))
    ax_a.set_yticklabels([f"{lab}  ({r['n_features']})"
                          for lab, r in zip(labels, rows)])
    ax_a.invert_yaxis()
    ax_a.set_xlim(0.68, 0.828)
    ax_a.set_xlabel('Test AUPRC')
    ax_a.set_title('Tested network features do not improve on four event counts in this design',
                   loc='left')
    ax_a.grid(axis='x', alpha=0.25, linestyle='--', lw=0.6)
    ax_a.set_axisbelow(True)

    # headline value only (section 2.5)
    focal_i = next(i for i, r in enumerate(rows) if r['config'] == FOCAL)
    ax_a.text(means[focal_i] + sds[focal_i] + 0.004, focal_i,
              f'{means[focal_i]:.3f}', va='center', ha='left',
              fontsize=ANNOT, fontweight='bold', color=C_FOCAL)

    # The two collapses, reported as raw AUPRC lost and as a share of the
    # model's gain over the random baseline (full - random), not as a share
    # of the Full Model value itself -- the manuscript text uses the same
    # gain-over-baseline denominator (Section 4.2).
    full = next(r['test_auprc_mean'] for r in rows
                if r['config'] == 'Full Model (All Features)')
    random_baseline = load_random_baseline()
    gain_over_random = full - random_baseline
    for i, r in enumerate(rows):
        if r['config'] in DROPS_HISTORY:
            drop_auprc = full - r['test_auprc_mean']
            pct_of_gain = drop_auprc / gain_over_random * 100
            ax_a.text(r['test_auprc_mean'] + r['test_auprc_std'] + 0.004, i,
                      f"{r['test_auprc_mean']:.3f}   \u2212{drop_auprc:.3f} AUPRC; "
                      f"\u2248{pct_of_gain:.0f}% of gain above random",
                      va='center', ha='left', fontsize=ANNOT, color=C_DROPS)

    ax_a.text(0.0, -0.30, f'higher = better   |   y-axis labels carry the feature '
                          f'count   |   mean of {n_seeds} seeds, bars = \u00b11 SD',
              transform=ax_a.transAxes, fontsize=ANNOT - 0.5, color=C_GREY)

    handles = [
        mpl.patches.Patch(color=C_FOCAL, label='Events only (recommended)'),
        mpl.patches.Patch(color=C_KEEPS, label='Keeps past-disappearance history'),
        mpl.patches.Patch(color=C_DROPS, label='History removed'),
    ]
    ax_a.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, -0.16),
                ncol=3, frameon=False, borderaxespad=0.0, handlelength=1.1,
                columnspacing=1.2, handletextpad=0.5)

    # ---------------- Panel b: zoom on the seven that keep history ----------------
    keep = [r for r in rows if r['config'] not in DROPS_HISTORY]
    yk = range(len(keep))
    kmeans = [r['test_auprc_mean'] for r in keep]
    ksds = [r['test_auprc_std'] for r in keep]
    kcolors = [C_FOCAL if r['config'] == FOCAL else C_KEEPS for r in keep]

    ax_b.errorbar(kmeans, list(yk), xerr=ksds, fmt='none',
                  ecolor=C_GREY, lw=1.0, capsize=2.5, zorder=2)
    ax_b.scatter(kmeans, list(yk), s=42, c=kcolors, zorder=3,
                 edgecolor='white', linewidth=0.6)
    ax_b.set_yticks(list(yk))
    ax_b.set_yticklabels([LABELS[r['config']] for r in keep])
    ax_b.invert_yaxis()
    spread = max(kmeans) - min(kmeans)
    ax_b.set_xlim(0.8035, 0.8175)
    ax_b.set_xticks([0.804, 0.808, 0.812, 0.816])
    ax_b.set_xlabel('Test AUPRC')
    ax_b.set_title('Within chance variation', loc='left')
    ax_b.grid(axis='x', alpha=0.25, linestyle='--', lw=0.6)
    ax_b.set_axisbelow(True)

    # How large an AUPRC difference arises on this test set by chance: the SD of
    # the paired permutation null from exp10, on the same events-only feature
    # set and the same 3,036 test sequences. 1.96 SD is the 95% band.
    null_sd = load_permutation_null_sd()
    band = 1.96 * null_sd
    lead = max(kmeans)
    ax_b.axvspan(lead - band, lead, color=C_FOCAL, alpha=0.08, zorder=0)
    ax_b.axvline(lead, color=C_FOCAL, lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax_b.text(lead - band * 0.52, 0.955,
              f'range {spread:.3f}\nvs chance \u00b1{band:.4f}',
              transform=ax_b.get_xaxis_transform(),
              ha='center', va='top', fontsize=ANNOT - 0.5, color=C_FOCAL,
              linespacing=1.35)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'ablation_chart.png', bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'ablation_chart.pdf', bbox_inches='tight')
    print(f"  Saved to {OUTPUT_DIR / 'ablation_chart.png'}")
    return fig


if __name__ == '__main__':
    create_ablation_chart()
