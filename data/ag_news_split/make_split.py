import sys, os, pandas as pd
src, out = sys.argv[1], sys.argv[2]
tr = pd.read_csv(os.path.join(src,'train.csv')); tr['_aug']=True
dv = pd.read_csv(os.path.join(src,'dev.csv')); te = pd.read_csv(os.path.join(src,'test.csv'))
other = pd.concat([dv,te], ignore_index=True); other['_aug']=False
alldf = pd.concat([tr, other], ignore_index=True).drop_duplicates(subset='content').reset_index(drop=True)
tot = len(alldf)
aug_cols = ['synonym_aug','back_translation','random_delete','insert_aug']
parts = {'train':[], 'dev':[], 'test':[]}
for lab, g in alldf.groupby('label'):
    n = len(g); n_tr = round(n*0.7); n_te = round(n*0.2)
    g = g.sample(frac=1, random_state=42)
    g = pd.concat([g[g._aug], g[~g._aug]])          # augmented rows first
    trn = g.iloc[:n_tr]; rest = g.iloc[n_tr:].sample(frac=1, random_state=42)
    parts['train'].append(trn); parts['test'].append(rest.iloc[:n_te]); parts['dev'].append(rest.iloc[n_te:])
train = pd.concat(parts['train']).sample(frac=1, random_state=42)
dev = pd.concat(parts['dev']).sample(frac=1, random_state=42)
test = pd.concat(parts['test']).sample(frac=1, random_state=42)
for c in aug_cols: train[c] = train[c].fillna(train['content'])
os.makedirs(out, exist_ok=True)
train[['label','content']+aug_cols].to_csv(os.path.join(out,'train.csv'), index=False)
dev[['label','content']].to_csv(os.path.join(out,'dev.csv'), index=False)
test[['label','content']].to_csv(os.path.join(out,'test.csv'), index=False)
for n,d in [('train',train),('dev',dev),('test',test)]:
    print(f"{n}: {len(d)} ({len(d)/tot:.1%}) {d.label.value_counts().sort_index().to_dict()}")
