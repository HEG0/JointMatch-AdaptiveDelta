"""End-to-end plumbing test that needs NO internet and NO GPU (runs in ~1-3 minutes on a laptop CPU).

It swaps BERT-base for a tiny randomly-initialised BERT, uses a small subset of AG News, and skips the
WordNet augmentation. The numbers it produces are meaningless; it only checks that training, the
adaptive disagreement weight, all file outputs and make_figures.py work.

    python smoke_test.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

import main                                   # noqa: E402  (changes cwd to this folder)
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers.models.bert.tokenization_bert import BasicTokenizer
import utils.dataloader as dl
from models.netgroup import NetGroup

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'experiment_smoke')
shutil.rmtree(OUT, ignore_errors=True)
tmp = tempfile.mkdtemp()

# ---- 1) small balanced subset of AG News ---------------------------------------------------------
src = os.path.join(HERE, '..', 'data', 'ag_news')
data_dir = os.path.join(tmp, 'ag_news')
os.makedirs(data_dir)
def balanced_subset(df, n_per_class):
    """n_per_class random rows of every class (written to work on old and new pandas versions)."""
    return df.sample(frac=1, random_state=0).groupby('label').head(n_per_class).reset_index(drop=True)


train = balanced_subset(pd.read_csv(os.path.join(src, 'train.csv')), 150)
dev = balanced_subset(pd.read_csv(os.path.join(src, 'dev.csv')), 30)
test = balanced_subset(pd.read_csv(os.path.join(src, 'test.csv')), 30)
train.to_csv(os.path.join(data_dir, 'train.csv'), index=False)
dev.to_csv(os.path.join(data_dir, 'dev.csv'), index=False)
test.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
shutil.copy(os.path.join(src, 'classes.txt'), data_dir)

# ---- 2) local tokenizer vocab + tiny BERT + identity augmentation --------------------------------------
basic = BasicTokenizer(do_lower_case=True)
words = set()
for df in (train, dev, test):
    for col in ('content', 'synonym_aug', 'back_translation'):
        if col in df:
            for t in df[col].astype(str):
                words.update(basic.tokenize(t))
vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'] + sorted(words)
vocab_file = os.path.join(tmp, 'vocab.txt')
open(vocab_file, 'w').write('\n'.join(vocab) + '\n')


class LocalTok:
    @staticmethod
    def from_pretrained(name):
        return BertTokenizer(vocab_file, do_lower_case=True)


class IdentityAug:
    def __init__(self, *a, **k): pass
    def augment(self, text): return [str(text)]


dl.BertTokenizer = LocalTok
dl.naw.SynonymAug = IdentityAug


def tiny_init_net(self, net_arch):
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=256, num_labels=self.n_classes,
                     initializer_range=0.5)   # large init -> peaked, net-specific outputs, so agree/disagree sets are non-empty
    return BertForSequenceClassification(cfg).to(self.device)


NetGroup.init_net = tiny_init_net

# ---- 3) run both adaptive variants, 2 seeds each ---------------------------------------------------------
from utils.adaptive_delta import VARIANTS   # noqa: E402

common = dict(dataset='ag_news', data_dir=data_dir, n_labeled_per_class=10, bs=8, ul_ratio=2, lr=1e-3,
              weight_u_loss=1, psl_threshold_h=0.7, adaptive_threshold=True, num_nets=2, cross_labeling=True,
              weight_disagreement='True', disagree_weight=0.9, ema_mode=False, ema_momentum=0.9,
              val_interval=5, early_stop_tolerance=20, max_step=60,
              delta_min=0.6, delta_max=0.9, delta_ema_momentum=0.9, delta_warmup_mass=5.0,   # fast EMA: the toy run is short
              keep_checkpoints=False, device_idx=0)
for name in ('adaptive_ema', 'paper_fixed'):
    main.multiRun(experiment_home=os.path.join(OUT, name), seeds_list=[0, 1], resume=False,
                  variant=name, **common, **VARIANTS[name])

# ---- 4) figures and tables ---------------------------------------------------------------------------
subprocess.check_call([sys.executable, os.path.join(HERE, 'make_figures.py'), '--experiment_home', OUT])

print('\nSMOKE TEST OK. Look at the files written to:', OUT)
for root_, _, files in os.walk(os.path.join(OUT, 'adaptive_ema', 'seed_0')):
    print('  seed_0 outputs:', sorted(files))
print('  analysis:', sorted(os.listdir(os.path.join(OUT, 'analysis'))))
print('  figures :', sorted(os.listdir(os.path.join(OUT, 'analysis', 'figures'))))
shutil.rmtree(tmp, ignore_errors=True)
