import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import TabTextAlign


def load_model(weights_path, emb_dim, num_classes, y_type, device, projector_mode="auto", residual_scale=1.0):
    checkpoint = torch.load(weights_path, map_location=device)
    # Remove 'module.' prefix
    new_state_dict = {}
    for k, v in checkpoint.items():
        new_k = k.replace('module.', '', 1) if k.startswith('module.') else k
        new_state_dict[new_k] = v

    if projector_mode == "auto":
        keys = set(new_state_dict)
        if any(k.startswith("w_tab.delta.") or k.startswith("w_text.delta.") for k in keys):
            projector_mode = "residual"
        elif any(k.startswith("w_tab.") or k.startswith("w_text.") for k in keys):
            projector_mode = "linear"
        else:
            projector_mode = "identity"

    model = TabTextAlign(
        emb_dim, num_classes, y_type,
        projector_mode=projector_mode,
        residual_scale=residual_scale,
    )
    # Only projection parameters are required for transformation. Ignoring the
    # auxiliary response head also permits transformation of checkpoints with
    # a different response-head shape.
    projection_state = {
        key: value
        for key, value in new_state_dict.items()
        if key == "logit_scale" or key.startswith("w_tab.") or key.startswith("w_text.")
    }
    model.load_state_dict(projection_state, strict=False)
    model.to(device)
    model.eval()
    return model


def project_all_tabular(dataset, projector, device, batch_size=1024):
    projector.eval()
    all_tab_proj = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Tabular (normalized) projection"):
            tab_dat, _, _ = batch
            tab_dat = tab_dat.to(device)
            tab_proj = projector(tab_dat).cpu().numpy()
            all_tab_proj.append(tab_proj)
    return np.concatenate(all_tab_proj, axis=0)


def project_and_save_text(dataset, projector, out_dir, split, device, batch_size=256):
    # Save into split dir by sample index (same as input)
    split_dir = os.path.join(out_dir, split)
    os.makedirs(split_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    projector.eval()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc=f"Text ({split})")):
            # This ensures text embeddings are stacked as [1, 77, 768]:
            _, _, text_emb = batch  # text_emb: (1, #tokens, emb_dim)
            text_emb = text_emb.squeeze(0).to(device)  # [77, 768]
            text_proj = projector(text_emb).cpu().numpy()
            np.save(os.path.join(split_dir, f"{idx}.npy"), text_proj)
