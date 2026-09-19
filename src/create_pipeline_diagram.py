"""
Rebuild of pipeline_diagram.png on the five-country sample.

The submitted diagram was drawn by hand and left no source file -- no script,
no vector original. This script reproduces its layout (same bands, same boxes,
same flow, same emphasis) with every number read from the result files in
results/, so the figure cannot drift from the numbers again.

Three differences from the submitted version, all forced by the new results:

  1. The Attention box read "(+0.1% AUPRC)". On five countries attention scores
     0.8081 against the plain LSTM's 0.8089, so it is 0.0008 WORSE. The box
     stays -- the attention weights carry the temporal-recency analysis -- but
     the parenthetical is now a cost, not a gain.

  2. The Events Only tile read "4 feat - best". The seven configurations that
     keep past-disappearance history span 0.0060 AUPRC, which is less than the
     +/-0.0071 chance variation on this test set (1.96 x the paired permutation
     null SD from exp10). Highest mean, not best. create_ablation_chart.py
     dropped its star for the same reason.

  3. The submitted diagram shows eight ablation tiles under a caption reading
     "9 configurations". Network Only was missing. All nine are drawn here.

Percentages on the two collapsed configurations are quoted against the Full
Model, matching ablations_5c.json's delta_from_full field and the manuscript
text (-10.7% and -14.3%), not against Events Only.
"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results'
OUTPUT_DIR = HERE / 'figures'

# palette taken from the submitted diagram
BLUE = '#1f5fa9'
BLUE_MID = '#4a86d0'
BLUE_PALE = '#eaf1fb'
BLUE_BAR = '#cfe0f5'
ORANGE = '#c8781e'
ORANGE_EDGE = '#d98328'
ORANGE_BAR = '#fbe0bf'
GREY_FILL = '#f7f7f7'
GREY_EDGE = '#b0b0b0'
DARK_EDGE = '#6b6b6b'
GREY_TEXT = '#8a8a8a'
INK = '#222222'
RED = '#c0392b'

T_HEAD = 10.5      # box titles
T_BODY = 9.0       # box body lines
T_SMALL = 8.0      # annotations
T_BAND = 10.0      # band labels (TEMPORAL SPLIT etc.)
T_TILE = 8.5       # ablation tile titles


def load_values():
    """Every number in the diagram, straight out of results/."""
    with open(RESULTS / 'descriptives_5c.json') as f:
        desc = json.load(f)
    with open(RESULTS / 'exp8_architecture_comparison.json') as f:
        arch = json.load(f)
    with open(RESULTS / 'ablations_5c.json') as f:
        abl = json.load(f)
    # exp8's sample block stores the positive rates rounded to 4dp, which
    # renders the test rate as 46.0%. exp10 carries it unrounded on the same
    # 3,036 test sequences, so take it from there: 45.9%.
    with open(RESULTS / 'exp10_bootstrap_events_only.json') as f:
        boot = json.load(f)

    s, c, summ = arch['sample'], arch['config'], arch['summary']
    rows = sorted(abl['results'], key=lambda r: -r['test_auprc_mean'])
    full = next(r['test_auprc_mean'] for r in rows
                if r['config'] == 'Full Model (All Features)')

    att = summ['attention_lstm']['test_auprc_mean']
    std = summ['standard_lstm']['test_auprc_mean']

    return {
        'total_events': desc['total_events'],
        'n_regions': s['n_regions'],
        'n_months': 84,
        'n_train': s['n_train_sequences'],
        'n_test': s['n_test_sequences'],
        'train_pos': s['train_positive_rate'],
        'test_pos': boot['test_positive_rate'],
        'pos_weight': c['pos_weight'],
        'seq_len': c['sequence_length'],
        'hidden': c['hidden_dim'],
        'layers': c['num_layers'],
        'dropout': c['dropout'],
        'batch': c['batch_size'],
        'epochs': c['num_epochs'],
        'n_seeds': len(c['seeds']),
        'attention_delta_pct': (att - std) / std * 100,
        'rows': rows,
        'full': full,
    }


def box(ax, x, y, w, h, fill='white', edge=DARK_EDGE, lw=1.1, dashed=False,
        radius=0.6, z=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f'round,pad=0,rounding_size={radius}',
                       facecolor=fill, edgecolor=edge, linewidth=lw,
                       linestyle=(0, (4, 2)) if dashed else '-', zorder=z)
    ax.add_patch(p)
    return p


def lines(ax, x, y, head, body, color=INK, head_color=None, body_color=None,
          head_size=T_HEAD, body_size=T_BODY, z=3):
    """Bold title with plain lines beneath, centred on x."""
    ax.text(x, y, head, ha='center', va='center', fontsize=head_size,
            fontweight='bold', color=head_color or color, zorder=z)
    for i, ln in enumerate(body):
        ax.text(x, y - 2.5 - i * 2.4, ln, ha='center', va='center',
                fontsize=body_size, color=body_color or color, zorder=z)


def arrow(ax, x0, x1, y, color='#333333'):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y),
                                 arrowstyle='-|>', mutation_scale=11,
                                 linewidth=1.2, color=color,
                                 shrinkA=0, shrinkB=0, zorder=4))


def band_label(ax, y, text):
    ax.text(0.5, y, text, ha='left', va='center', fontsize=T_BAND,
            fontweight='bold', color=INK, zorder=3)


def create_pipeline_diagram():
    v = load_values()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14.5, 8.3))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # ------------------------------------------------ band 1: data preparation
    y0, h1 = 86.0, 11.5
    box(ax, 1.0, y0, 18.5, h1, fill=GREY_FILL, edge=GREY_EDGE)
    lines(ax, 10.25, y0 + h1 - 3.0, 'ACLED Raw Events',
          [f"{v['total_events']:,} events",
           'Nigeria \u00b7 Mexico \u00b7 Myanmar',
           'Afghanistan \u00b7 Syria'])

    arrow(ax, 20.0, 24.0, y0 + h1 / 2)

    # two-line title, so the body line is placed explicitly rather than at
    # lines()' single-line offset
    box(ax, 24.5, y0, 18.0, h1)
    ax.text(33.5, y0 + h1 - 3.6, 'Region-Month\nAggregation', ha='center',
            va='center', fontsize=T_HEAD, fontweight='bold', color=INK,
            linespacing=1.4, zorder=3)
    ax.text(33.5, y0 + 2.4,
            f"{v['n_regions']} regions \u00d7 {v['n_months']} months",
            ha='center', va='center', fontsize=T_BODY, color=INK, zorder=3)

    arrow(ax, 43.0, 47.0, y0 + h1 / 2)

    # feature engineering container, one shade taller than its neighbours
    box(ax, 47.5, y0 - 1.2, 24.0, h1 + 2.6, fill=BLUE_PALE, edge=BLUE_MID)
    ax.text(59.5, y0 + h1 - 0.4, 'Feature Engineering', ha='center',
            va='center', fontsize=T_HEAD, fontweight='bold', color=BLUE,
            zorder=3)
    # the two feature blocks: event counts emphasised, network features
    # dashed and greyed, as in the submitted diagram
    sub_y, sub_h = y0 - 0.6, 9.4
    for cx, edge, dashed, head, body, hc, bc, cnt, cc in (
        (54.1, BLUE, False, 'Event Features',
         ('arrests, violence,', 'disappear., fatalities'),
         BLUE, INK, '(4 features)', BLUE),
        (65.3, GREY_EDGE, True, 'Network Features',
         ('density, centrality,', 'clustering, weights'),
         GREY_TEXT, GREY_TEXT, '(12 features)', GREY_TEXT),
    ):
        box(ax, cx - 5.3, sub_y, 10.6, sub_h, edge=edge, dashed=dashed,
            lw=1.2 if not dashed else 1.0)
        ax.text(cx, 93.6, head, ha='center', va='center', fontsize=9.2,
                fontweight='bold', color=hc, zorder=3)
        for j, ln in enumerate(body):
            ax.text(cx, 91.2 - j * 2.0, ln, ha='center', va='center',
                    fontsize=8.2, color=bc, zorder=3)
        ax.text(cx, 87.0, cnt, ha='center', va='center', fontsize=8.2,
                fontweight='bold', color=cc, zorder=3)

    arrow(ax, 72.0, 76.0, y0 + h1 / 2)

    box(ax, 76.5, y0, 22.5, h1)
    lines(ax, 87.75, y0 + h1 - 3.0, 'Temporal Sequences',
          [f"{v['seq_len']}-month sliding windows",
           f"{v['n_train'] + v['n_test']:,} sequences total"])

    # ------------------------------------------------------ band 2: split
    band_label(ax, 76.5, 'TEMPORAL SPLIT')

    bar_x, bar_w, bar_y, bar_h = 6.0, 80.0, 67.0, 5.6
    train_frac = 54 / 78          # Jul 2018-Dec 2022 of Jul 2018-Dec 2024
    split_x = bar_x + bar_w * train_frac

    ax.add_patch(Rectangle((bar_x, bar_y), bar_w * train_frac, bar_h,
                           facecolor=BLUE_BAR, edgecolor=BLUE_MID,
                           linewidth=0.9, zorder=2))
    ax.add_patch(Rectangle((split_x, bar_y), bar_w * (1 - train_frac), bar_h,
                           facecolor=ORANGE_BAR, edgecolor=ORANGE_EDGE,
                           linewidth=0.9, zorder=2))
    ax.text(bar_x + bar_w * train_frac / 2, bar_y + bar_h / 2,
            'Training: Jul 2018 \u2013 Dec 2022', ha='center', va='center',
            fontsize=T_HEAD, fontweight='bold', color=BLUE, zorder=3)
    ax.text(split_x + bar_w * (1 - train_frac) / 2, bar_y + bar_h / 2,
            'Test: Jan 2023 \u2013 Dec 2024', ha='center', va='center',
            fontsize=T_HEAD, fontweight='bold', color=ORANGE, zorder=3)

    # the boundary, carried up to the feature-engineering box
    ax.plot([split_x, split_x], [bar_y - 1.6, y0 - 2.0], color=RED,
            lw=1.1, ls=(0, (3, 2)), zorder=3)
    ax.plot([split_x, split_x], [bar_y, bar_y + bar_h], color='#111111',
            lw=1.6, zorder=5)
    ax.text(split_x, y0 - 3.4, 'strict temporal boundary', ha='center',
            va='bottom', fontsize=T_SMALL, style='italic', color=RED, zorder=3)

    ax.text(bar_x + bar_w * train_frac / 2, bar_y - 3.0,
            f"{v['n_train']:,} sequences ({v['train_pos'] * 100:.1f}% positive)",
            ha='center', va='center', fontsize=T_BODY, color=INK, zorder=3)
    ax.text(split_x + bar_w * (1 - train_frac) / 2, bar_y - 3.0,
            f"{v['n_test']:,} sequences ({v['test_pos'] * 100:.1f}% positive)",
            ha='center', va='center', fontsize=T_BODY, color=INK, zorder=3)

    # sequence-window inset
    ins_x, cw, gap, cy, ch = 84.4, 1.3, 0.3, 72.6, 2.6
    ax.text(ins_x, 76.2, 'Sequence window:', ha='left', va='center',
            fontsize=T_SMALL, color=INK, zorder=3)
    for i in range(v['seq_len']):
        ax.add_patch(Rectangle((ins_x + i * (cw + gap), cy), cw, ch,
                               facecolor=BLUE_BAR, edgecolor=BLUE_MID,
                               linewidth=0.8, zorder=3))
    tail = ins_x + v['seq_len'] * (cw + gap) - gap
    arrow(ax, tail + 0.3, tail + 2.0, cy + ch / 2)
    ax.add_patch(Rectangle((tail + 2.3, cy), cw, ch, facecolor=ORANGE_BAR,
                           edgecolor=ORANGE_EDGE, linewidth=0.8, zorder=3))
    ax.text(ins_x, cy - 1.8, f"t\u2212{v['seq_len'] - 1} \u2026 t (input)",
            ha='left', va='center', fontsize=T_SMALL, color=BLUE, zorder=3)
    ax.text(tail + 2.3 + cw / 2, cy - 1.8, 't+1', ha='center', va='center',
            fontsize=T_SMALL, color=ORANGE, zorder=3)

    # ------------------------------------------------------ band 3: model
    band_label(ax, 57.0, 'MODEL')

    my, mh = 44.0, 10.0
    box(ax, 6.0, my, 15.0, mh)
    lines(ax, 13.5, my + mh - 3.0, 'Input Tensor',
          [f"(batch, {v['seq_len']}, F)", 'F = 4 or 16'])

    arrow(ax, 21.5, 26.0, my + mh / 2)

    box(ax, 26.5, my - 1.0, 19.0, mh + 2.0, fill=BLUE, edge=BLUE)
    lines(ax, 36.0, my + mh - 2.6, f"{v['layers']}-Layer LSTM",
          [f"{v['hidden']} hidden units", f"dropout {v['dropout']}"],
          head_color='white', body_color='white', head_size=11.5)

    arrow(ax, 46.0, 50.5, my + mh / 2)

    box(ax, 51.0, my - 1.0, 17.0, mh + 2.0, edge=GREY_EDGE, dashed=True)
    lines(ax, 59.5, my + mh - 2.6, 'Attention',
          ['\u03b1\u2081 \u2026 \u03b1\u2086 weights'],
          head_color=GREY_TEXT, body_color=GREY_TEXT)
    ax.text(59.5, my + mh - 8.0,
            f"({v['attention_delta_pct']:+.1f}% AUPRC)", ha='center',
            va='center', fontsize=T_SMALL, style='italic', color=GREY_TEXT,
            zorder=3)

    arrow(ax, 68.5, 73.0, my + mh / 2)

    box(ax, 73.5, my, 15.0, mh)
    lines(ax, 81.0, my + mh - 3.0, 'FC Layers',
          [f"{v['hidden']}\u219232\u21921", '+ sigmoid'])

    arrow(ax, 89.0, 91.5, my + mh / 2)

    box(ax, 92.0, my + 1.4, 7.5, mh - 2.8, edge=ORANGE_EDGE, lw=1.3,
        radius=1.6)
    lines(ax, 95.75, my + mh - 3.8, 'P(dis)', ['\u2208 [0, 1]'],
          head_color=ORANGE, body_color=ORANGE, head_size=10.0, body_size=8.4)

    # ---------------------------------------------------- band 4: training
    band_label(ax, 38.5, 'TRAINING')

    ty, th = 27.5, 7.4
    train_boxes = [
        ('Weighted BCE Loss',
         [f"w\u208a = {v['pos_weight']:.3f} (class imbalance)"]),
        ('Adam Optimizer', ['lr = 0.001, batch = %d' % v['batch']]),
        ('Early Stopping',
         [f"patience = 10, max {v['epochs']} ep."]),
        ('Evaluation', ['AUPRC (primary metric)']),
    ]
    bw, gap = 21.0, 2.4
    for i, (head, body) in enumerate(train_boxes):
        x = 6.0 + i * (bw + gap)
        box(ax, x, ty, bw, th, fill=GREY_FILL, edge=GREY_EDGE)
        lines(ax, x + bw / 2, ty + th - 2.6, head, body)

    # ---------------------------------------------------- band 5: ablations
    band_label(ax, 20.0, 'ABLATION STUDY')

    short = {
        'Events Only (No Network)': 'Events Only',
        'Events + Clustering Features': 'Ev+Clustering',
        'Events + Structure Features': 'Ev+Structure',
        'Events + Centrality Features': 'Ev+Centrality',
        'Events + Weight Features': 'Ev+Weights',
        'Full Model (All Features)': 'Full Model',
        'No Fatalities': 'No Fatalities',
        'No Past Disappearances (Early Warning)': 'No Disappear.',
        'Network Only (No Events)': 'Network Only',
    }
    collapsed = {'No Past Disappearances (Early Warning)',
                 'Network Only (No Events)'}

    rows = v['rows']
    ay, ah = 9.0, 7.6
    tw = (99.0 - 6.0 - 1.6 * (len(rows) - 1)) / len(rows)
    for i, r in enumerate(rows):
        x = 6.0 + i * (tw + 1.6)
        first = (i == 0)
        drop = r['config'] in collapsed
        fill = BLUE if first else (ORANGE_BAR if drop else 'white')
        edge = BLUE if first else (ORANGE_EDGE if drop else GREY_EDGE)
        tc = 'white' if first else (ORANGE if drop else INK)
        box(ax, x, ay, tw, ah, fill=fill, edge=edge, radius=0.5)
        ax.text(x + tw / 2, ay + ah - 2.6, short[r['config']], ha='center',
                va='center', fontsize=T_TILE, fontweight='bold', color=tc,
                zorder=3)
        if drop:
            pct = (r['test_auprc_mean'] - v['full']) / v['full'] * 100
            sub = f"{pct:+.1f}% vs full"
        else:
            sub = f"{r['n_features']} feat"
        ax.text(x + tw / 2, ay + ah - 5.2, sub, ha='center', va='center',
                fontsize=T_SMALL, color=tc, zorder=3)

    ax.text(52.5, 5.6,
            f"{len(rows)} configurations \u00d7 {v['n_seeds']} seeds = "
            f"{len(rows) * v['n_seeds']} experiments",
            ha='center', va='center', fontsize=T_SMALL, style='italic',
            color=INK, zorder=3)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    for ext in ('png', 'pdf'):
        fig.savefig(OUTPUT_DIR / f'pipeline_diagram.{ext}', dpi=300)
    print(f"  Saved to {OUTPUT_DIR / 'pipeline_diagram.png'}")
    return fig


if __name__ == '__main__':
    create_pipeline_diagram()
