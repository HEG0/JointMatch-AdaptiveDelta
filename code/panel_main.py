"""Run JointMatch on AG News with the paper's fixed disagreement weight or the EMA-adaptive one.

Everything except the disagreement weight (delta) is identical to the original paper setup
(constant learning rate 1e-5, 10 labels per class, threshold 0.98, ...).

Examples
--------
python panel_main.py --variant adaptive_ema                  # delta from the EMA disagreement rate (5 seeds)
python panel_main.py --variant adaptive_ema --seeds 3        # a single seed
python panel_main.py --variant paper_fixed                   # control: the paper's fixed delta = 0.9
"""
import argparse

from main import multiRun
from utils.adaptive_delta import VARIANTS

parser = argparse.ArgumentParser()
parser.add_argument('--variant', required=True, choices=list(VARIANTS.keys()))
parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
parser.add_argument('--experiment_home', default='./experiment/ag_news',
                    help='results go to <experiment_home>/<variant>/seed_<n>/')
parser.add_argument('--device_idx', type=int, default=0)
parser.add_argument('--delta_min', type=float, default=0.6, help='lowest delta of the adaptive variant')
parser.add_argument('--delta_max', type=float, default=0.9, help='highest delta of the adaptive variant (paper value)')
parser.add_argument('--delta_ema_momentum', type=float, default=0.99,
                    help='EMA momentum (per unlabeled batch) of the agree/disagree counts')
parser.add_argument('--keep_checkpoints', action='store_true', help='keep the best-model checkpoints (large!)')
parser.add_argument('--no_resume', action='store_true', help='re-run seeds even if their results exist')
parser.add_argument('--max_step', type=int, default=100000)
args = parser.parse_args()

###########################
## AG_News  (hyper-parameters identical to the original repo)
n_labeled_per_class = 10
bs = 8
ul_ratio = 10
lr = 1e-5
weight_u_loss = 1
psl_threshold_h = 0.98
adaptive_threshold = True
num_nets = 2
cross_labeling = True
weight_disagreement = 'True'
disagree_weight = 0.9          # delta of the paper / of the 'paper_fixed' variant
ema_mode = False
ema_momentum = 0.9
val_interval = 25
early_stop_tolerance = 10
dataset = 'ag_news'

experiment_home = '%s/%s' % (args.experiment_home.rstrip('/'), args.variant)

multiRun(experiment_home=experiment_home, seeds_list=args.seeds, resume=not args.no_resume,
         variant=args.variant, device_idx=args.device_idx, dataset=dataset,
         n_labeled_per_class=n_labeled_per_class, bs=bs, ul_ratio=ul_ratio, lr=lr,
         weight_u_loss=weight_u_loss, psl_threshold_h=psl_threshold_h, adaptive_threshold=adaptive_threshold,
         num_nets=num_nets, cross_labeling=cross_labeling,
         weight_disagreement=weight_disagreement, disagree_weight=disagree_weight,
         ema_mode=ema_mode, ema_momentum=ema_momentum,
         val_interval=val_interval, early_stop_tolerance=early_stop_tolerance, max_step=args.max_step,
         delta_min=args.delta_min, delta_max=args.delta_max, delta_ema_momentum=args.delta_ema_momentum,
         keep_checkpoints=args.keep_checkpoints, **VARIANTS[args.variant])
