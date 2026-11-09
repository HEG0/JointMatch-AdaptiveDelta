"""Build all paper / slide figures and tables from the saved run outputs. No training needed.

    python make_figures.py                      # uses ./experiment/ag_news and whatever variants it finds
    python make_figures.py --variants paper_fixed adaptive_ema adaptive_inverse
    python make_figures.py --paper_baseline ../results/paper_baseline.json

Everything is written to <experiment_home>/analysis/ :
    figures/*.png (300 dpi, for slides)  and  figures/*.pdf (vector, for the paper)
    per_seed_results.csv, summary_table.csv, results_table.tex, significance.txt
"""
import argparse
import glob
import json
import os
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# colour-blind-safe palette (Okabe-Ito)
COLORS = {'paper_fixed': '#CC79A7', 'adaptive_ema': '#0072B2', 'adaptive_inverse': '#D55E00'}
LABELS = {'paper_fixed': r'Fixed $\delta$=0.9 (paper)',
          'adaptive_ema': r'EMA-adaptive $\delta$ (0.6-0.9)',
          'adaptive_inverse': r'Inverse EMA-adaptive $\delta$ (0.6-0.9)'}
SHORT = {'paper_fixed': 'Fixed (paper)', 'adaptive_ema': 'EMA-adaptive', 'adaptive_inverse': 'EMA-inverse'}

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13, 'legend.fontsize': 10.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'lines.linewidth': 2.0,
    'figure.dpi': 120, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'pdf.fonttype': 42, 'ps.fonttype': 42,   # editable text in vector figures
})


# ----------------------------------------------------------------------------- loading
def load_variant(home, variant):
    runs = []
    for d in sorted(glob.glob(os.path.join(home, variant, 'seed_*'))):
        fm_path = os.path.join(d, 'final_metrics.json')
        if not os.path.exists(fm_path):
            continue
        run = {'dir': d, 'fm': json.load(open(fm_path))}
        run['seed'] = run['fm']['seed']
        run['stats'] = pd.read_csv(os.path.join(d, 'training_statistics.csv'))
        run['trace'] = pd.read_csv(os.path.join(d, 'delta_trace.csv'))
        run['cm'] = pd.read_csv(os.path.join(d, 'confusion_matrix.csv'), index_col=0)
        runs.append(run)
    return runs


def mean_std_curve(runs, col):
    """mean / std over seeds at every evaluation step (only where >= 2 seeds still have data)."""
    df = pd.concat([r['stats'].set_index('step')[col].astype(float) for r in runs], axis=1)
    n = df.notna().sum(axis=1)
    keep = n >= min(2, len(runs))
    return df.mean(axis=1)[keep], df.std(axis=1, ddof=0)[keep].fillna(0)


def save(fig, out, name):
    os.makedirs(os.path.join(out, 'figures'), exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, 'figures', f'{name}.{ext}'))
    plt.close(fig)
    print('  saved figures/%s.{png,pdf}' % name)


def curve(ax, data, col, ylabel, scale=1.0, ylim=None):
    for v, runs in data.items():
        m, s = mean_std_curve(runs, col)
        ax.plot(m.index, m.values * scale, color=COLORS[v], label=LABELS[v])
        ax.fill_between(m.index, (m - s).values * scale, (m + s).values * scale, color=COLORS[v], alpha=0.18, lw=0)
    ax.set_xlabel('training step')
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)


# ----------------------------------------------------------------------------- figures
def fig_schedule(data, out):
    """delta_t and the EMA disagreement rate over training (mean +- std over seeds)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    curve(axes[0], data, 'delta_t', r'disagreement weight $\delta_t$')
    curve(axes[1], data, 'disagree_ema', 'EMA disagreement rate $r_t$\n(share of confident samples)', scale=1.0)
    for ax in axes:
        for v, runs in data.items():
            ax.axvline(np.mean([r['fm']['best_step'] for r in runs]), color=COLORS[v], ls=':', lw=1.5)
    axes[0].set_ylim(0.55, 0.95)
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color='gray', ls=':'))
    l.append('mean best step (same colour)')
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, out, 'fig1_delta_and_disagreement_ema')


def fig_val(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    curve(axes[0], data, 'acc_val', 'dev accuracy (%)', scale=100)
    curve(axes[1], data, 'f1_val', 'dev macro-F1 (%)', scale=100)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, out, 'fig2_dev_accuracy_f1')


def fig_psl(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    curve(axes[0], data, 'psl_total', 'confident pseudo-labels / eval interval')
    curve(axes[1], data, 'acc_psl', 'pseudo-label accuracy (%)', scale=100)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, out, 'fig3_pseudo_labels_quantity_quality')


def fig_diverge(data, out):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    curve(ax, data, 'dev_net_disagree_rate', 'dev prediction disagreement\nbetween the two nets (%)', scale=100)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, out, 'fig4_network_divergence')


def fig_agree(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for v, runs in data.items():
        for col, ls, lab in (('frac_agree', '-', 'agree'), ('frac_disagree', '--', 'disagree')):
            m, s = mean_std_curve(runs, col)
            axes[0].plot(m.index, m.values * 100, color=COLORS[v], ls=ls)
        for col, ls in (('psl_acc_agree', '-'), ('psl_acc_disagree', '--')):
            m, s = mean_std_curve(runs, col)
            axes[1].plot(m.index, m.values * 100, color=COLORS[v], ls=ls)
    axes[0].set_ylabel('share of unlabeled batch (%)')
    axes[1].set_ylabel('pseudo-label accuracy (%)')
    for a in axes:
        a.set_xlabel('training step')
    # legend: colour = variant, line style = subset
    handles = [plt.Line2D([], [], color=COLORS[v], label=SHORT[v]) for v in data]
    handles += [plt.Line2D([], [], color='k', ls='-', label='agree set'),
                plt.Line2D([], [], color='k', ls='--', label='disagree set')]
    axes[0].legend(handles=handles, frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, out, 'fig5_agree_vs_disagree_sets')


def _last_row_json(run, col):
    return np.array(json.loads(run['stats'][col].iloc[-1]), dtype=float)


def fig_classwise(data, out, class_names):
    k = len(class_names)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    width = 0.8 / len(data)
    x = np.arange(k)
    for j, (v, runs) in enumerate(data.items()):
        tot = np.stack([_last_row_json(r, 'cw_psl_total_accum') for r in runs])
        share = tot / tot.sum(1, keepdims=True) * 100
        acc = np.stack([_last_row_json(r, 'cw_psl_acc_accum') for r in runs]) * 100
        for ax, arr in ((axes[0], share), (axes[1], acc)):
            ax.bar(x + (j - (len(data) - 1) / 2) * width, np.nanmean(arr, 0), width * 0.95,
                   yerr=np.nanstd(arr, 0), capsize=3, color=COLORS[v], label=SHORT[v])
    axes[0].set_ylabel('share of all pseudo-labels (%)')
    axes[1].set_ylabel('pseudo-label accuracy (%)')
    axes[1].set_ylim(0, 100)
    for a in axes:
        a.set_xticks(x)
        a.set_xticklabels(class_names)
        a.grid(axis='x', alpha=0)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, out, 'fig6_classwise_pseudo_labels')


def fig_local_threshold(data, out, class_names):
    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 3.6), sharey=True, squeeze=False)
    for ax, (v, runs) in zip(axes[0], data.items()):
        arr = [np.stack([json.loads(s) for s in r['stats']['local_threshold']]) for r in runs]
        steps = runs[0]['stats']['step'].values
        L = min(len(a) for a in arr)
        mean = np.mean([a[:L] for a in arr], 0)
        for c, name in enumerate(class_names):
            ax.plot(steps[:L], mean[:, c], label=name)
        ax.set_title(SHORT[v])
        ax.set_xlabel('training step')
    axes[0][0].set_ylabel('class-wise local threshold\n(relative to max class)')
    axes[0][0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, out, 'fig7_local_thresholds')


def fig_confmat(data, out, class_names):
    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2), squeeze=False)
    for ax, (v, runs) in zip(axes[0], data.items()):
        norm = np.mean([r['cm'].values / r['cm'].values.sum(1, keepdims=True) for r in runs], 0)
        im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                ax.text(j, i, '%.2f' % norm[i, j], ha='center', va='center',
                        color='white' if norm[i, j] > 0.5 else 'black', fontsize=10)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=30)
        ax.set_yticklabels(class_names)
        ax.set_xlabel('predicted')
        ax.set_ylabel('true')
        ax.set_title(SHORT[v])
        ax.grid(False)
    fig.tight_layout()
    save(fig, out, 'fig8_confusion_matrix_mean_over_seeds')


def fig_final(data, out, paper):
    """Mean +- std (large marker) and every single seed (small dots). A dot plot is used instead of bars
    because the interesting differences are small and bars with a truncated axis would exaggerate them."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
    names = list(data.keys())
    for ax, key, title in ((axes[0], 'test_acc', 'test accuracy (%)'), (axes[1], 'test_f1', 'test macro-F1 (%)')):
        vals = [np.array([r['fm'][key] for r in data[v]]) * 100 for v in names]
        for i, (v, y) in enumerate(zip(names, vals)):
            ax.errorbar(i, y.mean(), yerr=y.std(), fmt='o', ms=9, capsize=6, lw=2, color=COLORS[v], zorder=3)
            ax.scatter(np.full(len(y), i) + np.linspace(-0.15, 0.15, len(y)), y, color=COLORS[v], alpha=0.45, s=22, zorder=2)
        pk = 'acc_mean' if key == 'test_acc' else 'f1_mean'
        extra = []
        if paper and paper.get(pk) is not None:
            ax.axhline(paper[pk], color='gray', ls='--', lw=1.5)
            ax.text(len(names) - 0.55, paper[pk], 'paper ', va='bottom', ha='right', color='gray', fontsize=10)
            extra = [np.array([paper[pk]])]
        allv = np.concatenate(vals + extra)
        pad = max(0.3, (allv.max() - allv.min()) * 0.25)
        ax.set_ylim(allv.min() - pad, allv.max() + pad)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([SHORT[v] for v in names], rotation=15, ha='right')
        ax.set_ylabel(title)
        ax.grid(axis='x', alpha=0)
    fig.tight_layout()
    save(fig, out, 'fig9_final_test_results')


# ----------------------------------------------------------------------------- tables
def tables(data, out, paper):
    rows = []
    for v, runs in data.items():
        for r in runs:
            fm = r['fm']
            rows.append({'variant': v, 'seed': fm['seed'], 'test_acc': 100 * fm['test_acc'], 'test_f1': 100 * fm['test_f1'],
                         'test_acc_net0': 100 * fm['test_acc_nets'][0], 'test_acc_net1': 100 * fm['test_acc_nets'][1],
                         'best_step': fm['best_step'], 'last_step': fm['last_step'],
                         'delta_at_best_step': fm['delta_at_best_step'],
                         'delta_mean_until_best': fm['delta_mean_until_best'], 'delta_mean_all': fm['delta_mean_all'],
                         'train_time_h': fm['train_time_sec'] / 3600})
    per_seed = pd.DataFrame(rows).sort_values(['variant', 'seed'])
    per_seed.to_csv(os.path.join(out, 'per_seed_results.csv'), index=False)

    summ = []
    for v, g in per_seed.groupby('variant', sort=False):
        summ.append({'variant': v, 'n_seeds': len(g),
                     'acc_mean': g.test_acc.mean(), 'acc_std': g.test_acc.std(ddof=0),
                     'f1_mean': g.test_f1.mean(), 'f1_std': g.test_f1.std(ddof=0),
                     'best_step_mean': g.best_step.mean(), 'delta_at_best_step_mean': g.delta_at_best_step.mean(),
                     'delta_mean_until_best': g.delta_mean_until_best.mean(),
                     'train_time_h_mean': g.train_time_h.mean()})
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(out, 'summary_table.csv'), index=False, float_format='%.4f')

    # LaTeX table (std = population std over seeds, as in the original JointMatch code)
    best_acc = summ.acc_mean.max()
    lines = [r'\begin{table}[t]', r'\centering', r'\begin{tabular}{lcc}', r'\toprule',
             r'Method & Accuracy (\%) & Macro-F1 (\%) \\', r'\midrule']
    if paper and paper.get('acc_mean') is not None:
        a = '%.2f' % paper['acc_mean'] + (r' $\pm$ %.2f' % paper['acc_std'] if paper.get('acc_std') is not None else '')
        f = ('%.2f' % paper['f1_mean'] + (r' $\pm$ %.2f' % paper['f1_std'] if paper.get('f1_std') is not None else '')) \
            if paper.get('f1_mean') is not None else '--'
        lines.append(r'JointMatch, fixed $\delta$=0.9 (paper) & %s & %s \\' % (a, f))
    for _, r in summ.iterrows():
        a = '%.2f $\\pm$ %.2f' % (r.acc_mean, r.acc_std)
        if abs(r.acc_mean - best_acc) < 1e-12:
            a = r'\textbf{%s}' % a
        name = SHORT[r.variant].replace('&', r'\&')
        lines.append(r'%s & %s & %.2f $\pm$ %.2f \\' % (name, a, r.f1_mean, r.f1_std))
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{AG News test results, mean $\pm$ std over %d seeds.}' % int(summ.n_seeds.max()),
              r'\label{tab:agnews-adaptive-delta}', r'\end{table}']
    open(os.path.join(out, 'results_table.tex'), 'w').write('\n'.join(lines) + '\n')

    # significance tests
    from scipy import stats
    txt = ['Paired tests across seeds (same seed = same labeled subset). With few seeds these tests have low power;',
           'treat p-values as indicative only.\n']
    for a, b in combinations(data.keys(), 2):
        A = per_seed[per_seed.variant == a].set_index('seed')
        B = per_seed[per_seed.variant == b].set_index('seed')
        common = A.index.intersection(B.index)
        for metric in ('test_acc', 'test_f1'):
            x, y = A.loc[common, metric].values, B.loc[common, metric].values
            line = '%s vs %s | %s | n=%d | mean diff (a-b) = %+.3f' % (a, b, metric, len(common), (x - y).mean())
            if len(common) >= 2:
                line += ' | paired t-test p=%.3f' % stats.ttest_rel(x, y).pvalue
                try:
                    line += ' | Wilcoxon p=%.3f' % stats.wilcoxon(x, y).pvalue
                except ValueError:
                    pass
            txt.append(line)
    if paper and paper.get('acc_mean') is not None:
        txt.append('\nOne-sample t-test of each variant\'s accuracy against the paper mean (%.2f); indicative only:' % paper['acc_mean'])
        for v in data:
            y = per_seed[per_seed.variant == v].test_acc.values
            if len(y) >= 2:
                txt.append('%s | mean %.2f | p=%.3f' % (v, y.mean(), stats.ttest_1samp(y, paper['acc_mean']).pvalue))
    open(os.path.join(out, 'significance.txt'), 'w').write('\n'.join(txt) + '\n')
    print('\n'.join(txt))
    print('\n', summ.round(3).to_string(index=False))


# ----------------------------------------------------------------------------- main
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment_home', default=os.path.join(HERE, 'experiment', 'ag_news'))
    ap.add_argument('--variants', nargs='+', default=None)
    ap.add_argument('--paper_baseline', default=os.path.join(HERE, '..', 'results', 'paper_baseline.json'))
    ap.add_argument('--classes', default=os.path.join(HERE, '..', 'data', 'ag_news', 'classes.txt'))
    args = ap.parse_args()

    variants = args.variants or [v for v in COLORS if os.path.isdir(os.path.join(args.experiment_home, v))]
    data = {v: load_variant(args.experiment_home, v) for v in variants}
    data = {v: r for v, r in data.items() if r}
    if not data:
        raise SystemExit('no finished runs found under %s' % args.experiment_home)
    print('variants / finished seeds:', {v: [r['seed'] for r in rs] for v, rs in data.items()})

    paper = json.load(open(args.paper_baseline)) if os.path.exists(args.paper_baseline) else None
    class_names = [l.strip() for l in open(args.classes) if l.strip()] if os.path.exists(args.classes) \
        else list(next(iter(data.values()))[0]['cm'].columns)

    out = os.path.join(args.experiment_home, 'analysis')
    os.makedirs(out, exist_ok=True)
    print('writing to', out)
    tables(data, out, paper)
    fig_schedule(data, out)
    fig_val(data, out)
    fig_psl(data, out)
    fig_diverge(data, out)
    fig_agree(data, out)
    fig_classwise(data, out, class_names)
    fig_local_threshold(data, out, class_names)
    fig_confmat(data, out, class_names)
    fig_final(data, out, paper)
    print('done')
