import argparse, json, os, shutil
from concurrent.futures import ThreadPoolExecutor
import joblib
import numpy as np
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from .common import *

def data():
 frozen_protocol(); dest=ROOT/'data';dest.mkdir(parents=True,exist_ok=True)
 if (ROOT/'data_manifest.json').exists():check_test();log('Keeping verified existing data');return
 test_names=['X_num_test.npy','X_cat_test.npy','y_test.npy','text_test.json']
 hashes={n:sha(TEST_SOURCE/n) for n in test_names}
 # Require original pool and current test membership/order to agree.
 for n in test_names:
  if n.endswith('.npy'):np.testing.assert_array_equal(np.load(TRAIN_SOURCE/n),np.load(TEST_SOURCE/n))
  else:assert json.loads((TRAIN_SOURCE/n).read_text())==json.loads((TEST_SOURCE/n).read_text())
 for n in test_names:shutil.copy2(TEST_SOURCE/n,dest/n)
 write_json(ROOT/'test_fingerprints.json',hashes)
 n=len(np.load(TRAIN_SOURCE/'y_train.npy'));assert n==PROTOCOL['n_train']+PROTOCOL['n_val']+PROTOCOL['n_cal'], 'Expected the original 61813-row pool, not wine_review3_wval/train'
 perm=np.random.default_rng(PROTOCOL['split_seed']).permutation(n)
 nc,nv=PROTOCOL['n_cal'],PROTOCOL['n_val']
 indices={'cal':perm[:nc],'val':perm[nc:nc+nv],'train':perm[nc+nv:]}
 assert np.unique(np.concatenate(list(indices.values()))).size==n
 assert len(np.load(dest/'y_test.npy'))==PROTOCOL['n_test']
 texts=json.loads((TRAIN_SOURCE/'text_train.json').read_text());assert len(texts)==n
 manifest={'source':str(TRAIN_SOURCE),'test_source':str(TEST_SOURCE),'split_seed':PROTOCOL['split_seed'],'sources':{},'splits':{},'test':check_test()}
 for stem in ['X_num','X_cat','y']:
  p=TRAIN_SOURCE/f'{stem}_train.npy';a=np.load(p,allow_pickle=False);assert len(a)==n
  manifest['sources'][str(p)]=sha(p)
  for s,idx in indices.items():np.save(dest/f'{stem}_{s}.npy',a[idx])
 for s,idx in indices.items():
  np.save(dest/f'source_indices_{s}.npy',idx)
  write_json(dest/f'text_{s}.json',[texts[int(i)] for i in idx])
  manifest['splits'][s]={'n':len(idx),'index_sha256':sha(dest/f'source_indices_{s}.npy')}
 # Snapshot upstream code whose behavior this run depends on.
 paths=list(Path(__file__).resolve().parent.rglob('*.py'))+[REPO/'scripts/observed_residual_pi.py',REPO/'scripts/run_wine_pi.py',REPO/'configs/wine_pi_protocol.json']
 pack=REPO/'src/tabsynfnn'
 paths += [pack/'tabsyn/vae/main_unitabsyn.py',pack/'tabsyn/vae/model_fnn.py',pack/'tabsyn/main_pm.py',pack/'tabsyn/model_pm.py',pack/'tabsyn/sample_pm.py']
 snapshot=ROOT/'source_snapshot';snapshot.mkdir(exist_ok=True)
 records={}
 for i,p in enumerate(paths):
  target=snapshot/f'{i:02d}_{p.name}';shutil.copy2(p,target);records[str(p)]={'sha256':sha(p),'snapshot':str(target)}
 write_json(ROOT/'upstream_sources.json',records);write_json(ROOT/'data_manifest.json',manifest)
 log(f"Created disjoint train={len(indices['train'])}, val={len(indices['val'])}, calibration={len(indices['cal'])}; original test unchanged")

def transformed(split,fit=False):
 out=ROOT/'preprocessed';out.mkdir(exist_ok=True)
 x=np.load(ROOT/'data'/f'X_num_{split}.npy').astype(np.float64)
 c=np.load(ROOT/'data'/f'X_cat_{split}.npy').astype(str)
 if fit:
  assert split=='train'
  means=np.nanmean(x,axis=0);assert np.isfinite(means).all()
  x=np.where(np.isnan(x),means,x)
  qt=QuantileTransformer(output_distribution='normal',n_quantiles=max(1,min(1000,len(x)//30)),subsample=int(1e9),random_state=0)
  # Match the package's quantile preprocessing: fitted only on training covariates.
  qt.fit(x)
  vocab=[sorted(np.unique(c[:,j]).tolist()) for j in range(c.shape[1])]
  maps=[{v:k for k,v in enumerate(values)} for values in vocab]
  scaler=StandardScaler().fit(read_y('train').reshape(-1,1))
  pp={'numeric_means':means,'quantile':qt,'category_maps':maps,'categories':[len(v)+1 for v in vocab],'response_scaler':scaler}
  joblib.dump(pp,out/'fitted_train_only.joblib')
  ydir=ROOT/'shared/response';ydir.mkdir(parents=True,exist_ok=True);joblib.dump(scaler,ydir/'y_scaler.joblib')
 else:pp=joblib.load(out/'fitted_train_only.joblib')
 xx=pp['quantile'].transform(np.where(np.isnan(x),pp['numeric_means'],x)).astype(np.float32)
 cc=np.column_stack([np.array([m.get(v,len(m)) for v in c[:,j]],dtype=np.int64) for j,m in enumerate(pp['category_maps'])])
 assert np.isfinite(xx).all()
 np.save(out/f'num_{split}.npy',xx);np.save(out/f'cat_{split}.npy',cc)
 if split in ['train','val']:
  np.save(ROOT/'shared/response'/f'y_{split}_scaled.npy',pp['response_scaler'].transform(read_y(split).reshape(-1,1)).astype(np.float32))
 write_json(out/f'{split}_transform.json',{'n':len(xx),'unknown_counts':[int((cc[:,j]==len(m)).sum()) for j,m in enumerate(pp['category_maps'])],'fit_on_train_only':True})

def preprocess_fit():
 transformed('train',True);transformed('val')
 log('Preprocessing fitted exclusively on train; transformed train/validation')

def pack_text(splits):
 out=ROOT/'text_banks';out.mkdir(exist_ok=True)
 for split in splits:
  p=out/f'{split}.npy';marker=out/f'{split}.complete.json'
  if marker.exists():log(f'Keeping packed {split} text');continue
  if split=='test':ids=np.arange(PROTOCOL['n_test']);src=TEXT/'test'
  else:ids=np.load(ROOT/'data'/f'source_indices_{split}.npy');src=TEXT/'train'
  # Frozen source CLIP embeddings; no response or fitted statistics used.
  def load_one(i):
   a=np.load(src/f'{int(i)}.npy',allow_pickle=False).squeeze().astype(np.float32)
   assert a.shape==(77,768) and np.isfinite(a).all(),(split,int(i),a.shape)
   return a
  bank=np.lib.format.open_memmap(p,mode='w+',dtype=np.float32,shape=(len(ids),77,768))
  with ThreadPoolExecutor(max_workers=8) as pool:
   for start in range(0,len(ids),256):
    for offset,a in enumerate(pool.map(load_one,ids[start:start+256])):bank[start+offset]=a
    if start%4096==0:log(f'Packing {split} text: {start}/{len(ids)}')
  bank.flush();del bank
  write_json(marker,{'n':len(ids),'source':str(src),'shape':[len(ids),77,768],'index_sha256':sha(ROOT/'data'/f'source_indices_{split}.npy') if split!='test' else 'identity','file_sha256':sha(p)})
  log(f'Packed {split} text')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['data','preprocess','text_fit','text_heldout','heldout']);a=ap.parse_args()
 if a.stage=='data':data()
 elif a.stage=='preprocess':preprocess_fit()
 elif a.stage=='text_fit':pack_text(['train','val'])
 elif a.stage=='text_heldout':pack_text(['cal','test'])
 else:
  variant=os.environ['WINE_VARIANT']
  assert (ROOT/variant/'selection.json').exists(),'Select the model before transforming heldout features'
  check_test()
  for s in ['cal','test']:transformed(s)
if __name__=='__main__':main()
