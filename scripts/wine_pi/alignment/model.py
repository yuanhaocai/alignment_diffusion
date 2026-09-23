"""
Functions and classes for models and losses.
"""
import numpy as np
import torch


class IdentityProjector(torch.nn.Module):
    def forward(self, x):
        return x


class ResidualProjector(torch.nn.Module):
    def __init__(self, emb_dim=768, residual_scale=1.0, zero_init=True):
        super().__init__()
        self.delta = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        self.residual_scale = residual_scale
        if zero_init:
            torch.nn.init.zeros_(self.delta.weight)

    def forward(self, x):
        return x + self.residual_scale * self.delta(x)


def build_projector(mode, emb_dim=768, residual_scale=1.0):
    if mode == "linear":
        return torch.nn.Linear(emb_dim, emb_dim, bias=False)
    if mode == "identity_init":
        projector = torch.nn.Linear(emb_dim, emb_dim, bias=False)
        torch.nn.init.eye_(projector.weight)
        return projector
    if mode == "residual":
        return ResidualProjector(emb_dim=emb_dim, residual_scale=residual_scale)
    if mode == "identity":
        return IdentityProjector()
    raise ValueError(f"Unknown projector mode: {mode}")


class TabTextAlign(torch.nn.Module):
    def __init__(
            self, emb_dim=768, num_classes=None, y_type=None,
            projector_mode="linear", residual_scale=1.0
        ):
        super().__init__()
        self.projector_mode = projector_mode
        self.residual_scale = residual_scale
        self.w_tab = build_projector(projector_mode, emb_dim, residual_scale)
        self.w_text = build_projector(projector_mode, emb_dim, residual_scale)
        # Learnable log-temperature (initial value matches CLIP's 0.07)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if y_type == "categorical":
            assert num_classes is not None
            out_dim = num_classes
        else:
            out_dim = 1
        self.head = torch.nn.Linear(emb_dim, out_dim)
        # Separate response head on the text projection. Lets the supervised
        # loss flow into w_text directly; the original code only supervised
        # w_tab via self.head, so w_text learned only through the contrastive
        # loss. Used only when weight_sup_loss_text > 0; harmless otherwise.
        self.head_text = torch.nn.Linear(emb_dim, out_dim)

    def forward(self, tab_emb, text_emb):
        tab_proj = self.w_tab(tab_emb)
        text_proj = self.w_text(text_emb)
        return tab_proj, text_proj, self.logit_scale


def compute_combined_cosine_sim(tab, text):
    """
    tab: [B, 768], text: [B, 77, 768]
    For each pair (i, j) in the batch: 
    Compute 1/77 * sum_k cosine(tab[i], text[j][k]) for k=0..76
    """
    # B, N_tok, D = text.shape
    tab_norm = tab / (tab.norm(dim=1, keepdim=True) + 1e-8)
    text_norm = text / (text.norm(dim=2, keepdim=True) + 1e-8)
    
    # [B,1,768] vs [1,B,77,768] -> broadcasting
    # We'll build [B, B, 77]. S[i,j,k] = cosine(tab[i], text[j,k])
    # This can be computed as: (tab_norm[i] * text_norm[j,k]).sum(-1).
    # Let’s expand tab_norm: [B,1,1,768], text_norm: [1,B,77,768]
    tab_exp = tab_norm[:, None, None, :]        # [B, 1, 1, 768]
    text_exp = text_norm[None, :, :, :]         # [1, B, 77, 768]
    cos_sim = (tab_exp * text_exp).sum(-1)      # [B, B, 77]
    s = cos_sim.mean(-1)                        # [B, B]  (mean over tokens)
    return s   # for i-th tab and j-th text, s[i,j] (cosine sim between i-th tab and all tokens of j-th text)


def clip_loss(sim, logit_scale):
    """
    sim: [B, B], sim[i, j] is similarity of tab[i], text[j]
    """
    sim = sim * logit_scale.exp()
    B = sim.size(0)
    labels = torch.arange(B, device=sim.device)
    # cross entropy loss over rows
    loss_text = torch.nn.functional.cross_entropy(sim, labels)
    # cross entropy loss over columns (for symmetry)
    loss_tab = torch.nn.functional.cross_entropy(sim.t(), labels)
    return (loss_text + loss_tab) / 2


def unwrap_model(model):
    """
    Recursively unwraps a distributed or Accelerate-wrapped model to get to the original nn.Module.
    """
    # Official DDP and Accelerate both wrap with a `.module` attribute
    while hasattr(model, "module"):
        model = model.module
    return model
