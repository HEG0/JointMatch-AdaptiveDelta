# JointMatch with an EMA-adaptive disagreement weight (AG News)

A modified copy of [HenryPengZou/JointMatch](https://github.com/HenryPengZou/JointMatch)
(*JointMatch: A Unified Approach for Diverse and Collaborative Pseudo-Labeling to Semi-Supervised Text Classification*,
Zou & Caragea, EMNLP 2023), restricted to **AG News**.

**The only algorithmic change:** in the original code the weight on "disagreement" samples, `disagree_weight` (δ), is the
constant 0.9. Here δ can instead be computed from an exponential moving average (EMA) of how often the two networks
disagree. Everything else (constant learning rate 1e-5, thresholds, losses, early stopping, hyper-parameters, data) is the
original setup.

## The adaptive rule

The loss is unchanged: `weight = δ · 1[nets disagree on confidence] + (1 − δ) · 1[nets agree on confidence]`.
Only δ is now a function of training:

```
A_t = m·A_{t-1} + (1−m)·(#unlabeled samples in the batch that BOTH nets are confident about)   (agree)
D_t = m·D_{t-1} + (1−m)·(#samples that exactly ONE net is confident about)                      (disagree)
r_t = D_t / (A_t + D_t)                         EMA disagreement rate among confident samples
δ_t = δ_min + (δ_max − δ_min) · (1 − r_t)       δ_min = 0.6, δ_max = 0.9, m = 0.99
```

* Few disagreements (r_t → 0): the nets are alike, which is the collapse the disagreement weight is meant to prevent, so
  δ → 0.9 (the paper's value).
* Many disagreements: the nets are already diverse and agreement samples are the more reliable pseudo-labels, so the
  weight moves toward them (δ decreases, never below 0.6; the paper requires δ in (0.5, 1)).
* Until the EMA has seen enough confident samples, δ = 0.9, so training starts exactly like the paper.

This rule is a design choice, not something from the paper, and it has not been tuned.

### Variants (`--variant`)

| name | δ |
|---|---|
| `adaptive_ema` | EMA rule above |
| `paper_fixed` | constant 0.9 (the paper; use it as your control) |

## Setup

```bash
conda create -n jointmatch python=3.8 -y && conda activate jointmatch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch     # pick a CUDA build if you have a GPU
pip install -r requirements.txt
python code/setup_offline.py     # ONE time, with internet: caches bert-base-uncased + NLTK WordNet data
```

## 1. Check the install (no internet / GPU needed, 1-3 min on a CPU)
```bash
cd code && python smoke_test.py
```
Uses a tiny random BERT on a small subset: numbers are meaningless, but it checks training, the adaptive δ, all outputs
and `make_figures.py`. Output goes to `code/experiment_smoke/` (git-ignored).

## 2. Run the experiments
```bash
cd code
python panel_main.py --variant adaptive_ema      # 5 seeds, sequential
python panel_main.py --variant paper_fixed       # control, same 5 seeds
```
Single seed: add `--seeds 3`. Finished seeds are skipped on re-run (resume). On a SLURM cluster (e.g. Penn State Roar)
edit the `<PLACEHOLDERS>` in `code/slurm/run_roar.sbatch` and run `mkdir -p slurm_logs && sbatch code/slurm/run_roar.sbatch`.

Runs take roughly 3 to 7 hours each on a laptop CPU (the number of steps depends on early stopping).
The same seed gives the same labeled subset in both variants, so the comparison is paired by seed.

## 3. Outputs
Per run, in `code/experiment/ag_news/<variant>/seed_<n>/`:

| file | content |
|---|---|
| `config.json` | every hyper-parameter, seed, resolved settings |
| `training_statistics.csv` | per dev evaluation: `delta_t`, `disagree_ema`, dev acc/F1 (ensemble and each net), pseudo-label count and accuracy, share and accuracy of the agree / disagree sets, dev disagreement between the two nets, class-wise lists (JSON strings) |
| `delta_trace.csv` | δ and the EMA rate at every training step |
| `final_metrics.json` | test acc / macro-F1 (ensemble and each net), best step, δ at the best step and mean δ until it, runtime |
| `confusion_matrix.csv`, `per_class_metrics.csv`, `test_predictions.csv` | test-set details (predictions allow e.g. McNemar tests) |

Checkpoints are deleted after the final test evaluation (`--keep_checkpoints` keeps them; >1 GB per run).

## 4. Figures and tables
```bash
cd code && python make_figures.py
```
Writes `experiment/ag_news/analysis/`: figures as `.png` (300 dpi, slides) and `.pdf` (vector, paper):
1. `fig1_delta_and_disagreement_ema`: δ_t and r_t over training
2. `fig2_dev_accuracy_f1`
3. `fig3_pseudo_labels_quantity_quality`
4. `fig4_network_divergence`
5. `fig5_agree_vs_disagree_sets`
6. `fig6_classwise_pseudo_labels`, 7. `fig7_local_thresholds`
8. `fig8_confusion_matrix_mean_over_seeds`
9. `fig9_final_test_results` (mean ± std, every seed as a dot, the paper's number as a dashed line)

and tables `per_seed_results.csv`, `summary_table.csv`, `results_table.tex`, `significance.txt` (paired tests; with 5 seeds
these have low power).

`results/paper_baseline.json` holds the paper's AG News accuracy (88.39) for the dashed line and the table row. Verify it
against the paper's results table and fill in the std / F1 if you have them.

## Notes
* Tested: the full pipeline on CPU with a tiny BERT and a recent Python/PyTorch/pandas. The real `bert-base-uncased`
  training loop was run on a laptop CPU with Python 3.8 and the pinned versions during development of an earlier
  version of the δ rule; the EMA rule itself has only been run in the smoke test so far.
* The original repository does not state a license. This copy keeps the original code structure and credits; check the
  upstream repo before redistributing publicly.

## Citation
```bibtex
@inproceedings{zou2023jointmatch,
    title={JointMatch: A Unified Approach for Diverse and Collaborative Pseudo-Labeling to Semi-Supervised Text Classification},
    author={Zou, Henry and Caragea, Cornelia},
    booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
    pages={7290--7301},
    year={2023}
}
```
Data and code are borrowed from [SAT](https://github.com/declare-lab/SAT) and [USB](https://github.com/microsoft/Semi-supervised-learning) by the original authors.
