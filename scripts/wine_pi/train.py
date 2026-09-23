import argparse, functools, importlib, json, os, shutil, sys
from types import SimpleNamespace
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset
from .common import *

# All changes below are in this standalone process. Installed package files stay unchanged.
def quiet_tqdm(*args,**kwargs):
 from tqdm import tqdm
 kwargs['disable']=True
 return tqdm(*args,**kwargs)

def pp():return joblib.load(ROOT/'preprocessed/fitted_train_only.joblib')
def proc(s):
 return np.load(ROOT/'preprocessed'/f'num_{s}.npy'),np.load(ROOT/'preprocessed'/f'cat_{s}.npy')

def vae():
 from tabsynfnn.tabsyn.vae import main_unitabsyn as m
 cfg=PROTOCOL['vae'];seed_all(cfg['seed']);out=ROOT/'shared/vae';out.mkdir(parents=True,exist_ok=True)
 def train_validation_only(*args,**kwargs):
  xn,xc=proc('train');vn,vc=proc('val');p=pp()
  yn=np.load(ROOT/'shared/response/y_train_scaled.npy');yv=np.load(ROOT/'shared/response/y_val_scaled.npy')
  return (xn,None,vn),(xc,None,vc),(yn,None,yv),p['categories'],xn.shape[1]
 m.preprocess=train_validation_only;m.tqdm=quiet_tqdm
 # The release VAE uses index 2 for validation; the test slot stays None.
 m.vae_train_unitabsyn(str(ROOT/'data'),str(out),latent_dim=cfg['latent_dim'],d_token=cfg['d_token'],task_type='regression',train_target='dat',device_ids=[0],epochs=cfg['epochs'],batch_size=cfg['batch_size'],lr=cfg['lr'])
 if (out/'test_z_dat.npy').exists():(out/'test_z_dat.npy').rename(out/'legacy_validation_z_dat.npy')
 write_json(out/'training_protocol.json',{'training_split':'train','scheduler_split':'val','checkpoint_selection':'minimum training reconstruction loss, matching existing VAE code','fitted_preprocessing_sha256':sha(ROOT/'preprocessed/fitted_train_only.joblib'),'model_sha256':sha(out/'model_dat.pt')})

def encoder():
 from tabsynfnn.tabsyn.vae.model_fnn import Encoder_model_fnn,Model_VAE_fnn
 p=pp();cfg=PROTOCOL['vae'];nnum=np.load(ROOT/'preprocessed/num_train.npy',mmap_mode='r').shape[1];seq=nnum+len(p['categories'])
 model=Model_VAE_fnn(nnum,p['categories'],cfg['d_token'],seq,cfg['latent_dim'],bias=True).cuda()
 parallel=torch.nn.DataParallel(model,device_ids=[0]);parallel.load_state_dict(torch.load(ROOT/'shared/vae/model_dat.pt',map_location='cuda'))
 enc=Encoder_model_fnn(nnum,p['categories'],cfg['d_token'],seq,cfg['latent_dim']).cuda();enc.load_weights(model);enc.eval()
 return enc

def encode(splits):
 seed_all(PROTOCOL['vae']['seed']);enc=encoder();out=ROOT/'shared/vae'
 with torch.no_grad():
  for s in splits:
   xn,xc=proc(s);bank=[]
   for start in range(0,len(xn),1024):
    z=enc(torch.from_numpy(xn[start:start+1024]).float().cuda(),torch.from_numpy(xc[start:start+1024]).long().cuda())
    bank.append(z.cpu().numpy())
   z=np.concatenate(bank);assert z.shape==(len(xn),768) and np.isfinite(z).all()
   np.save(out/f'{s}_z_dat.npy',z);log(f'Encoded frozen VAE {s}: {z.shape}')
 if 'train' in splits:
  z=torch.from_numpy(np.load(out/'train_z_dat.npy')).float()
  joblib.dump({'mean':z.mean(0).numpy(),'std':z.std(0).numpy()},out/'train_latent_normalization.joblib')

class PackedAlignmentDataset(Dataset):
 def __init__(self,text_path,train_z_dat_path,y_path,y_type,split):
  assert split in ['train','val']
  stats=joblib.load(ROOT/'shared/vae/train_latent_normalization.joblib')
  z=np.load(ROOT/'shared/vae'/f'{split}_z_dat.npy')
  self.z=torch.from_numpy((z-stats['mean'])/(stats['std']+1e-6)).float()
  self.y=torch.from_numpy(read_y(split).astype(np.float32))
  self.text=np.load(ROOT/'text_banks'/f'{split}.npy',mmap_mode='r')
 def __len__(self):return len(self.y)
 def __getitem__(self,i):return self.z[i],self.y[i],np.array(self.text[i],copy=True)

class SingleGPUAccelerator:
 is_main_process=True
 def __init__(self):self.device=torch.device('cuda')
 def prepare(self,*args):
  out=tuple(x.to(self.device) if isinstance(x,torch.nn.Module) else x for x in args)
  return out if len(out)>1 else out[0]
 def backward(self,loss):loss.backward()
 def gather_for_metrics(self,x):return x.detach().reshape(-1)

def cosine_sim(tab,text):
 # Exact algebraic equivalent of averaging pairwise token cosines.
 a=tab/(tab.norm(dim=1,keepdim=True)+1e-8)
 b=text/(text.norm(dim=2,keepdim=True)+1e-8)
 return a@b.mean(dim=1).T

def alignment_imports():
 from .alignment import main, model, evaluate
 return main,model,evaluate

def align():
 cfg=PROTOCOL['alignment'];seed_all(cfg['seed']);m,model,ev=alignment_imports()
 m.Dataset_text_tabular=PackedAlignmentDataset;m.Accelerator=SingleGPUAccelerator;m.tqdm=quiet_tqdm
 m.compute_combined_cosine_sim=cosine_sim;ev.compute_combined_cosine_sim=cosine_sim
 args=SimpleNamespace(text_path=str(ROOT/'text_banks'),train_z_dat_path=str(ROOT/'shared/vae'),y_path=str(ROOT/'data'),save_dir=str(ROOT/'full/alignment'),batch_size=cfg['batch_size'],emb_dim=768,y_num_classes=5,y_class_weight=None,lr=cfg['lr'],n_epochs=cfg['epochs'],num_workers=2,eval_interval=5,weight_clip_loss=1.,weight_sup_loss=1.,weight_sup_loss_text=0.,text_pool_for_sup='mean',projector_mode='residual',residual_scale=.1,eval_use_train_set=False,patience=25,y_type='continuous')
 m.main(args)


def alignment_model():
 _,m,_=alignment_imports();model=m.TabTextAlign(emb_dim=768,num_classes=5,y_type='continuous',projector_mode='residual',residual_scale=.1).cuda()
 model.load_state_dict(torch.load(ROOT/'full/alignment/models/tabtext_align.pt',map_location='cuda'));model.eval();return model

def project(splits):
 model=alignment_model();stats=joblib.load(ROOT/'shared/vae/train_latent_normalization.joblib');out=ROOT/'full/conditioning';out.mkdir(exist_ok=True,parents=True)
 with torch.no_grad():
  for s in splits:
   z=np.load(ROOT/'shared/vae'/f'{s}_z_dat.npy');z=(z-stats['mean'])/(stats['std']+1e-6)
   result=[]
   for start in range(0,len(z),1024):result.append(model.w_tab(torch.from_numpy(z[start:start+1024]).float().cuda()).cpu().numpy())
   np.save(out/f'{s}_tab.npy',np.concatenate(result))
   texts=np.load(ROOT/'text_banks'/f'{s}.npy',mmap_mode='r');pooled=[]
   for start in range(0,len(texts),256):
    t=torch.from_numpy(np.array(texts[start:start+256],copy=True)).float().cuda()
    # Project tokens first, then take exactly the original diffusion's token mean.
    pooled.append(model.w_text(t).mean(dim=1).cpu().numpy())
   a=np.concatenate(pooled);assert a.shape==(len(z),768) and np.isfinite(a).all();np.save(out/f'{s}_text_mean.npy',a)
   log(f'Projected {s} using validation-selected alignment; cached exact diffusion mean pooling')
 write_json(out/'provenance.json',{'alignment_checkpoint':str(ROOT/'full/alignment/models/tabtext_align.pt'),'sha256':sha(ROOT/'full/alignment/models/tabtext_align.pt'),'pooling':'Token projection followed by arithmetic mean; normalization flags all false, identical to original diffusion input computation.'})

VARIANT=None
OVERRIDE_SPLIT=None
SAMPLE_MODE=False
class PackedDiffusionDataset(Dataset):
 def __init__(self,y_emb_path,emb_alignment_path,y_type,split,vae_dat_path=None,text_path=None,image_path=None,use_tab=True,use_text=True,use_image=False,**kwargs):
  split=OVERRIDE_SPLIT or split
  assert split in ['train','val','cal','test']
  if VARIANT=='full':
   self.tab=torch.from_numpy(np.load(ROOT/'full/conditioning'/f'{split}_tab.npy')).float()
   self.text=torch.from_numpy(np.load(ROOT/'full/conditioning'/f'{split}_text_mean.npy')).float()
  else:
   self.tab=torch.from_numpy(np.load(ROOT/'shared/vae'/f'{split}_z_dat.npy')).float();self.text=torch.zeros((len(self.tab),1))
  if SAMPLE_MODE:self.y=torch.zeros((len(self.tab),1))
  else:
   assert split=='train';self.y=torch.from_numpy(np.load(ROOT/'shared/response/y_train_scaled.npy')).float()
  self.img=torch.zeros((len(self.tab),1));assert len(self.y)==len(self.tab)
 def __len__(self):return len(self.tab)
 def __getitem__(self,i):return self.y[i],self.tab[i],self.text[i],self.img[i]

def diffusion(variant):
 global VARIANT,OVERRIDE_SPLIT,SAMPLE_MODE
 VARIANT=variant;OVERRIDE_SPLIT=None;SAMPLE_MODE=False
 from tabsynfnn.tabsyn import main_pm as m
 cfg=PROTOCOL[variant];seed_all(cfg['seed']);m.PredictiveModelingDataset=PackedDiffusionDataset;m.tqdm=quiet_tqdm
 out=ROOT/variant/'diffusion'
 m.diffusion_train_pm(y_emb_path=str(ROOT/'shared/response'),emb_alignment_path=str(ROOT/'full/conditioning') if variant=='full' else None,diffusion_path=str(out),y_type='continuous',use_tab=True,use_text=variant=='full',use_image=False,vae_dat_path=str(ROOT/'shared/vae') if variant!='full' else None,epochs=cfg['epochs'],batch_size=cfg['batch_size'],lr=cfg['lr'],d_in=1,dim_t=cfg['dim_t'],cond_dim=768,text_pooling='mean',normalize_tab_cond=False,normalize_text_tokens=False,normalize_text_pooled=False,normalize_projected_cond=False,alignment_mix_alpha=1.,early_stopping_thrshd=500,clip_grad=True,clip_grad_max_norm=2.,save_milestone_epochs='250,500,750,1000',use_wandb=False,resume=(out/'checkpoint.pt').exists(),checkpoint_path=str(out/'checkpoint.pt'),checkpoint_interval=10,save_checkpoint=True,seed=cfg['seed'])


def sample(variant,split,filename='selected_model.pt',M=200,out=None):
 global VARIANT,OVERRIDE_SPLIT,SAMPLE_MODE
 VARIANT=variant;OVERRIDE_SPLIT=split;SAMPLE_MODE=True
 from tabsynfnn.tabsyn import sample_pm as m
 m.PredictiveModelingDataset=PackedDiffusionDataset;m.tqdm=quiet_tqdm
 # Continuous sampling only needs the fitted response scaler. Avoid legacy preprocessing entirely.
 m.preprocess=lambda *args,**kwargs:(None,None,None,None,None,None,None)
 seed_all(PROTOCOL['sample_seeds'][variant][split])
 out=out or ROOT/variant/f'sampling_{split}_m{M}'
 m.diffusion_repeated_sample_pm(data_path=str(ROOT/'data'),diffusion_path=str(ROOT/variant/'diffusion'),sampling_path=str(out),batch_size=1024,steps=PROTOCOL['sampling_steps'],num_repeats=M,use_val=split in ['val','cal'],base_seed=PROTOCOL['sample_seeds'][variant][split],model_filename=filename)
 write_json(out/'semantic_split.json',{'actual_split':split,'M':M,'seed':PROTOCOL['sample_seeds'][variant][split],'outcomes_loaded_for_sampling':False,'model_sha256':sha(ROOT/variant/'diffusion'/filename)})


def select(variant):
 candidates=[];seen={};y=read_y('val');root=ROOT/variant
 for filename in PROTOCOL['diffusion_candidates']:
  p=root/'diffusion'/filename
  if not p.exists():continue
  h=sha(p)
  if h in seen:
   candidates.append({'filename':filename,'sha256':h,'same_as':seen[h]});continue
  seen[h]=filename;out=root/'validation_selection'/filename.replace('.pt','')
  sample(variant,'val',filename,PROTOCOL['validation_selection_M'],out)
  bank=np.load(out/'y_test_all.npy');assert bank.shape==(PROTOCOL['validation_selection_M'],len(y)) and np.isfinite(bank).all()
  rmse=float(np.sqrt(np.mean((y-bank.astype(float).mean(0))**2)))
  candidates.append({'filename':filename,'sha256':h,'rmse_raw':rmse})
 scored=[c for c in candidates if 'rmse_raw' in c];assert scored
 chosen=min(scored,key=lambda c:c['rmse_raw'])
 shutil.copy2(root/'diffusion'/chosen['filename'],root/'diffusion/selected_model.pt')
 write_json(root/'selection.json',{'criterion':PROTOCOL['diffusion_selection'],'split':'val','n':len(y),'M':PROTOCOL['validation_selection_M'],'candidates':candidates,'selected':chosen,'selected_model_sha256':sha(root/'diffusion/selected_model.pt')})
 log(f'{variant} selected {chosen}')


def selfcheck():
 seed_all(991);_,model,_=alignment_imports()
 a=torch.randn(5,768,dtype=torch.float64,device='cuda',requires_grad=True)
 b=torch.randn(5,77,768,dtype=torch.float64,device='cuda',requires_grad=True)
 original=model.compute_combined_cosine_sim(a,b);fast=cosine_sim(a,b)
 torch.testing.assert_close(original,fast,atol=1e-12,rtol=1e-10)
 weights=torch.randn_like(fast)
 g1=torch.autograd.grad((original*weights).sum(),(a,b),retain_graph=True);g2=torch.autograd.grad((fast*weights).sum(),(a,b))
 for x,y in zip(g1,g2):torch.testing.assert_close(x,y,atol=1e-12,rtol=1e-10)
 from tabsynfnn.tabsyn.model_pm import MLPDiffusion_pm
 net=MLPDiffusion_pm(1,64,cond_dim=768,use_tab=True,use_text=True,use_image=False,text_pooling='mean',normalize_tab_cond=False,normalize_text_tokens=False,normalize_text_pooled=False,normalize_projected_cond=False).cuda()
 a=torch.randn(5,768,device='cuda');b=torch.randn(5,77,768,device='cuda');x=torch.randn(5,1,device='cuda');noise=torch.randn(5,device='cuda');img=torch.zeros(5,1,device='cuda')
 v1=net(x,noise,a,b,img);v2=net(x,noise,a,b.mean(1),img)
 torch.testing.assert_close(v1,v2,atol=1e-7,rtol=1e-6)
 # Verify unknown category indices are inside the reserved embedding range.
 p=pp()
 for s in ['train','val']:
  xn,xc=proc(s)
  for j,n in enumerate(p['categories']):assert xc[:,j].min()>=0 and xc[:,j].max()<n
 write_json(ROOT/'selfcheck.json',{'cosine_value_and_gradient_equivalence':True,'diffusion_cached_mean_equivalence':True,'training_validation_category_bounds':True,'test_immutability':check_test()})
 log('Self-check passed: optimized computations preserve the model; test files identical')


def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['vae','encode_fit','align','project_fit','diffusion','select','encode_heldout','project_heldout','sample','selfcheck']);ap.add_argument('--variant',choices=['full','tabular_only']);ap.add_argument('--split',choices=['cal','test']);a=ap.parse_args();frozen_protocol()
 if a.stage=='vae':vae()
 elif a.stage=='encode_fit':encode(['train','val'])
 elif a.stage=='align':align()
 elif a.stage=='project_fit':project(['train','val'])
 elif a.stage=='diffusion':diffusion(a.variant)
 elif a.stage=='select':select(a.variant)
 elif a.stage=='encode_heldout':encode(['cal','test'])
 elif a.stage=='project_heldout':project(['cal','test'])
 elif a.stage=='sample':
  selection=json.loads((ROOT/a.variant/'selection.json').read_text());assert sha(ROOT/a.variant/'diffusion/selected_model.pt')==selection['selected_model_sha256'];sample(a.variant,a.split,M=PROTOCOL['M'])
 else:selfcheck()
if __name__=='__main__':main()
