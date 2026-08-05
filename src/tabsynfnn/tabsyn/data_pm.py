# Portions adapted from amazon-science/TabSyn (Apache-2.0) and modified.
import os
from typing import Literal
import numpy as np
from torch.utils.data import Dataset


class PredictiveModelingDataset(Dataset):
    def __init__(
            self,
            y_emb_path,
            emb_alignment_path, 
            y_type: Literal['continuous', 'categorical'],
            split: Literal['train', 'test', 'val'],
            vae_dat_path = None,
            text_path = None,
            image_path = None,
            use_tab: bool = True,
            use_text: bool = True,
            use_image: bool = False,
            alignment_mix_alpha: float = 1.0,
            raw_vae_dat_path = None,
            raw_text_path = None,
        ):
        """
        Dataset for predictive modeling diffusion training. Reads categorical y, tabular predictor
        embeddings transformed by alignment methods, and text embeddings transformed by alignment methods.
        """
        assert split in ['train', 'test', 'val']
        self.split = split
        self.use_tab = use_tab
        self.use_text = use_text
        self.use_image = use_image
        self.alignment_mix_alpha = float(alignment_mix_alpha)
        if self.alignment_mix_alpha < 0.0 or self.alignment_mix_alpha > 1.0:
            raise ValueError("alignment_mix_alpha must be in [0, 1].")
        self.raw_text_path = raw_text_path
        
        # y must exist
        if y_type == "categorical":
            y_emb = np.load(os.path.join(y_emb_path, f'{split}_z_y.npy'))
            self.y_emb = np.squeeze(y_emb, axis=1)
        elif y_type == "continuous":
            self.y_emb = np.load(os.path.join(y_emb_path, f'y_{split}_scaled.npy'))

        if emb_alignment_path:
            if self.use_tab:
                self.z_proj = np.load(os.path.join(emb_alignment_path, f'{split}_z_dat_proj.npy'))
                if self.alignment_mix_alpha < 1.0:
                    if raw_vae_dat_path is None:
                        raise ValueError("raw_vae_dat_path is required when alignment_mix_alpha < 1 and use_tab=True.")
                    try:
                        raw_z_proj = np.load(os.path.join(raw_vae_dat_path, f'{split}_z_dat.npy'))
                    except FileNotFoundError:
                        raw_z_proj = np.load(os.path.join(raw_vae_dat_path, f'X_num_{split}_scaled.npy'))
                    self.z_proj = (
                        (1.0 - self.alignment_mix_alpha) * raw_z_proj
                        + self.alignment_mix_alpha * self.z_proj
                    ).astype(np.float32)
            else:
                self.z_proj = None
            self.text_path = os.path.join(emb_alignment_path, "text_proj")
            self.image_path = os.path.join(emb_alignment_path, "image_proj") # for future placeholder
            if self.use_text and self.alignment_mix_alpha < 1.0 and self.raw_text_path is None:
                raise ValueError("raw_text_path is required when alignment_mix_alpha < 1 and use_text=True.")
        else:
            if self.use_tab:
                try:
                    self.z_proj = np.load(os.path.join(vae_dat_path, f'{split}_z_dat.npy'))
                except:
                    self.z_proj = np.load(os.path.join(vae_dat_path, f'X_num_{split}_scaled.npy'))
            else:
                self.z_proj = None
            self.text_path = text_path
            self.image_path = image_path

    def __len__(self):
        return self.y_emb.shape[0]
    
    def __getitem__(self, idx):
        """
        Output: 
            y_emb: [B, d_token or 1]; tab_dat (z_proj): [B, 768]; text: [B, 77, 768]
        """
        if self.use_tab:
            z_proj = self.z_proj[idx,]
        else:
            z_proj = float('nan')
        if self.use_text:
            text_emb_proj = np.load(os.path.join(self.text_path, self.split, f'{idx}.npy')).squeeze()
            if self.alignment_mix_alpha < 1.0:
                raw_text = np.load(os.path.join(self.raw_text_path, self.split, f'{idx}.npy')).squeeze()
                text_emb_proj = (
                    (1.0 - self.alignment_mix_alpha) * raw_text
                    + self.alignment_mix_alpha * text_emb_proj
                ).astype(np.float32)
        else:
            text_emb_proj = float('nan')
        if self.use_image:
            img_emb_proj = np.load(os.path.join(self.image_path, self.split, f'{idx}.npy')).squeeze()
        else:
            img_emb_proj = float('nan')
        
        return self.y_emb[idx,], z_proj, text_emb_proj, img_emb_proj
