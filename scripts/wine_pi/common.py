from pathlib import Path
import hashlib, json, os, random, time
import numpy as np

import sys
REPO=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('WINE_RUN_ROOT', REPO/'artifacts/wine_prediction_interval')).resolve()
TRAIN_SOURCE=Path(os.environ.get('WINE_POOL_DIR', REPO/'data/wine_review3')).resolve()
TEST_SOURCE=Path(os.environ.get('WINE_TEST_DIR', TRAIN_SOURCE)).resolve()
TEXT=Path(os.environ.get('WINE_TEXT_DIR', REPO/'embeddings/wine_review3_text_clip_embd')).resolve()
ALIGN=Path(__file__).resolve().parent/'alignment'
PYTHON=sys.executable
PROTOCOL=json.loads((REPO/'configs/wine_pi_protocol.json').read_text())

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''): h.update(x)
 return h.hexdigest()

def write_json(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)

def log(msg): print(time.strftime('%Y-%m-%d %H:%M:%S'),msg,flush=True)
def seed_all(seed):
 import torch
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.set_num_threads(4)

def frozen_protocol():
 p=ROOT/'protocol.json'
 if p.exists(): assert json.loads(p.read_text())==PROTOCOL,'Protocol changed after initialization'
 else: write_json(p,PROTOCOL)
 return PROTOCOL

def check_test():
 initial=json.loads((ROOT/'test_fingerprints.json').read_text());out={}
 for name,expected in initial.items():
  source=sha(TEST_SOURCE/name);copy=sha(ROOT/'data'/name)
  assert source==expected and copy==expected,f'Test file changed: {name}'
  out[name]={'sha256':expected,'source_unchanged':True,'copy_identical':True}
 return out

def read_y(split):return np.load(ROOT/'data'/f'y_{split}.npy',allow_pickle=False).reshape(-1).astype(float)
