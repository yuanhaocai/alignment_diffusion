# Portions adapted from amazon-science/TabSyn (Apache-2.0) and modified.
"""
Built upon 'model.py', but for predictive modeling. To be used in 'main_pm.py'.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import PositionalEmbedding
from .diffusion_utils import SIGMA_MAX, SIGMA_MIN, S_churn, S_min, S_max, S_noise, rho, randn_like


class MLPDiffusion_pm(nn.Module):
    def __init__(
            self, 
            d_in, 
            dim_t, 
            cond_dim=768, 
            text_dim=768, 
            image_dim=512,
            use_tab: bool = True,
            use_text: bool = True,
            use_image: bool = False,  
            text_pooling: str = "mean",
            normalize_tab_cond: bool = True,
            normalize_text_tokens: bool = False,
            normalize_text_pooled: bool = True,
            normalize_projected_cond: bool = False,
        ):
        super().__init__()
        if text_pooling not in {"mean", "first"}:
            raise ValueError(f"Unknown text_pooling={text_pooling!r}; expected 'mean' or 'first'.")
        self.dim_t = dim_t
        self.tab_proj = nn.Linear(cond_dim, dim_t)
        self.text_proj = nn.Linear(text_dim, dim_t)
        self.img_proj = nn.Linear(image_dim, dim_t)
        self.use_tab = use_tab
        self.use_text = use_text
        self.use_image = use_image
        self.text_pooling = text_pooling
        self.normalize_tab_cond = normalize_tab_cond
        self.normalize_text_tokens = normalize_text_tokens
        self.normalize_text_pooled = normalize_text_pooled
        self.normalize_projected_cond = normalize_projected_cond
        concat_modalities = []
        concat_in_features = d_in
        if self.use_tab:
            concat_modalities.append("tab")
            concat_in_features += dim_t
        if self.use_text:
            concat_modalities.append("text")
            concat_in_features += dim_t
        if self.use_image:
            concat_modalities.append("img")
            concat_in_features += dim_t
        self.concat_modalities = concat_modalities
        self.proj = nn.Linear(concat_in_features, dim_t)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )
        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

    @staticmethod
    def _l2_normalize(x):
        return F.normalize(x, p=2, dim=-1, eps=1e-8)

    def _pool_text(self, text_emb):
        if text_emb.dim() == 2:
            textemb = text_emb
        elif text_emb.dim() == 3:
            token_mask = text_emb.norm(dim=-1).gt(0).to(text_emb.dtype)
            if self.normalize_text_tokens:
                text_emb = self._l2_normalize(text_emb)
            if self.text_pooling == "mean":
                denominator = token_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                textemb = (
                    text_emb * token_mask.unsqueeze(-1)
                ).sum(dim=1) / denominator
            elif self.text_pooling == "first":
                textemb = text_emb[:, 0, :]
            else:
                raise RuntimeError(f"Unsupported text_pooling={self.text_pooling!r}.")
        else:
            raise ValueError(f"text_emb must have shape [batch, dim] or [batch, tokens, dim], got {text_emb.shape}.")

        if self.normalize_text_pooled:
            textemb = self._l2_normalize(textemb)
        return textemb

    def forward(self, x, noise_labels, tab_pred_emb, text_emb, img_emb):
        """
        Args:
            x: categorical response embedding, of shape [batch_size, d_in], where d_in usually is just 
                the number of levels.
            tab_pred_emb: [batch_size, 768]
            text_emb: [batch_size, 77, 768]. When `self.use_text` is False, this is not used.
            img_emb: [batch_size, 512].
            noise_labels: [batch]
        """
        inputs = [x]
        if self.use_tab:
            if self.normalize_tab_cond:
                tab_pred_emb = self._l2_normalize(tab_pred_emb)
            tabemb = self.tab_proj(tab_pred_emb)   # [batch, dim_t]
            if self.normalize_projected_cond:
                tabemb = self._l2_normalize(tabemb)
            inputs.append(tabemb)
        if self.use_text:
            textemb = self._pool_text(text_emb)    # [batch, text_dim]
            textemb = self.text_proj(textemb)      # [batch, dim_t]
            if self.normalize_projected_cond:
                textemb = self._l2_normalize(textemb)
            inputs.append(textemb)
        if self.use_image:
            imgemb = self.img_proj(img_emb)        # [batch, dim_t]
            if self.normalize_projected_cond:
                imgemb = self._l2_normalize(imgemb)
            inputs.append(imgemb)
        h = torch.cat(inputs, dim=-1)              # [batch, d_in + enabled * dim_t]
        h = self.proj(h)                           # [batch, dim_t]

        # Add time/noise embedding
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        emb = self.time_embed(emb)                                  # [batch, dim_t]
        h = h + emb

        # Pass to MLP
        return self.mlp(h)
    

class Precond_pm(nn.Module):
    def __init__(self,
        denoise_fn,
        hid_dim,
        sigma_min = 0,                # Minimum supported noise level.
        sigma_max = float('inf'),     # Maximum supported noise level.
        sigma_data = 0.5,             # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.denoise_fn_F = denoise_fn

    def forward(self, x, sigma, tab_pred, text, img):
        x = x.to(torch.float32)

        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        F_x = self.denoise_fn_F((x_in).to(dtype), c_noise.flatten(), tab_pred, text, img)

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    

class EDMLoss_pm:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, hid_dim = 100, gamma=5, opts=None):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hid_dim = hid_dim
        self.gamma = gamma
        self.opts = opts

    def __call__(self, denoise_fn, data, tab_pred, text, img):
        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(data) * sigma.unsqueeze(1)
        D_yn = denoise_fn(data + n, sigma, tab_pred, text, img)
    
        loss = weight.unsqueeze(1) * ((D_yn - data) ** 2)
        return loss


class Model_pm(nn.Module):
    def __init__(
            self, denoise_fn, hid_dim=None, P_mean=-1.2, P_std=1.2, sigma_data=0.5,
            class_weights=None, normalize_class_weights=True
        ):
        super().__init__()
        self.denoise_fn_D = Precond_pm(denoise_fn, hid_dim)
        self.loss_fn = EDMLoss_pm(P_mean, P_std, sigma_data, hid_dim=hid_dim, gamma=5, opts=None)
        self.class_weights = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        self.normalize_class_weights = normalize_class_weights

    def _labels_from_y_embedding(self, x):
        if x.dim() != 2:
            raise ValueError(f"Expected y embedding with shape [batch, classes], got {tuple(x.shape)}.")
        if self.class_weights is not None and x.shape[1] != len(self.class_weights):
            raise ValueError(
                f"class_weights has length {len(self.class_weights)}, but y embedding has width {x.shape[1]}."
            )
        return x.argmax(dim=-1)

    def forward(self, x, tab_pred, text, img):
        loss = self.loss_fn(self.denoise_fn_D, x, tab_pred, text, img).mean(-1)
        if self.class_weights is not None:
            labels = self._labels_from_y_embedding(x)
            sample_weights = self.class_weights.to(device=loss.device, dtype=loss.dtype)[labels]
            if self.normalize_class_weights:
                sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-8)
            loss = loss * sample_weights
        return loss.mean()


def sample_pm(net, num_samples, dim, num_steps, tab_pred, text, img, device):
    """
    Args:
        y: provide a tensor of shape (num_samples, dim_labels)
    """
    latents = torch.randn([num_samples, dim], device=device)

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)

    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float32) * t_steps[0]

    def sample_step(net, num_steps, i, t_cur, t_next, x_next, tab_pred, text, img):
        x_cur = x_next
        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        # Euler step.

        t_hat_in = torch.full((x_hat.shape[0], 1), t_hat).to(x_hat.device)
        denoised = net(x_hat, t_hat_in, tab_pred, text, img).to(torch.float32)

        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            t_next_in = torch.full((x_hat.shape[0], 1), t_next).to(x_hat.device)
            denoised = net(x_next, t_next_in, tab_pred, text, img).to(torch.float32)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            print(f'Sample timestep {num_steps-i:4d}', end='\r')
            x_next = sample_step(net, num_steps, i, t_cur, t_next, x_next, tab_pred, text, img)

    return x_next
