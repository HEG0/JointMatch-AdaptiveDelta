"""Download everything the code needs at run time, so jobs can run on compute nodes without internet.

Run ONCE on a machine with internet access (e.g. the Roar login node), inside the conda env:

    python setup_offline.py

It caches
  * the NLTK data used by nlpaug's WordNet synonym augmentation (applied on-the-fly by the data collator)
  * bert-base-uncased (tokenizer + weights) in the Hugging Face cache (~/.cache/huggingface, or $HF_HOME)
"""
import nltk

for pkg in ['wordnet', 'omw-1.4', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'punkt', 'punkt_tab']:
    ok = nltk.download(pkg, quiet=True)
    print('nltk %-34s %s' % (pkg, 'ok' if ok else 'not available (fine if your nltk version does not use it)'))

from transformers import BertModel, BertTokenizer

BertTokenizer.from_pretrained('bert-base-uncased')
BertModel.from_pretrained('bert-base-uncased')
print('bert-base-uncased cached.')

# sanity check: the augmenter used by the data collator must work
import nlpaug.augmenter.word as naw
print('augmentation test:', naw.SynonymAug(aug_src='wordnet', aug_p=0.05).augment('The company reported strong quarterly earnings.'))
print('\nSetup complete. Jobs can now run offline (HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1).')
