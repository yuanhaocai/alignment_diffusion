# Portions adapted from amazon-science/TabSyn (Apache-2.0) and modified.
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import betaprime


#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

randn_like=torch.randn_like

# some hyperparameters
SIGMA_MIN=0.002
SIGMA_MAX=80
rho=7
S_churn= 1
S_min=0
S_max=float('inf')
S_noise=1


def sample(net, num_samples, dim, num_steps=50, y=None, device='cuda:0'):
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
    y = y.to(device)

    def sample_step(net, num_steps, i, t_cur, t_next, x_next, y):

        x_cur = x_next
        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        # Euler step.

        t_hat_in = torch.full((x_hat.shape[0], 1), t_hat).to(x_hat.device)
        denoised = net(x_hat, t_hat_in, y).to(torch.float32)

        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            t_next_in = torch.full((x_hat.shape[0], 1), t_next).to(x_hat.device)
            denoised = net(x_next, t_next_in, y).to(torch.float32)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            print(f'Sample timestep {num_steps-i:4d}', end='\r')
            x_next = sample_step(net, num_steps, i, t_cur, t_next, x_next, y)

    return x_next


def sample_nndataparallel(net, num_samples, dim, num_steps=50, y=None, device='cuda:0'):
    """
    See `sample`.
    """
    latents = torch.randn([num_samples, dim], device=device)

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)

    sigma_min = max(SIGMA_MIN, net.module.sigma_min)
    sigma_max = min(SIGMA_MAX, net.module.sigma_max)

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.module.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float32) * t_steps[0]
    y = y.to(device)

    def sample_step(net, num_steps, i, t_cur, t_next, x_next, y):
        x_cur = x_next
        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.module.round_sigma(t_cur + gamma * t_cur) 
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        #---TEST---
        # print(x_hat.shape, y.shape)
        #------

        # Euler step.
        t_hat_in = torch.full((x_hat.shape[0], 1), t_hat).to(x_hat.device)
        denoised = net(x_hat, t_hat_in, y).to(torch.float32)

        #---TEST---
        # print(t_hat_in.shape, denoised.shape, t_hat.shape)
        #------

        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        t_next_in =  torch.full((x_hat.shape[0], 1), t_next).to(x_hat.device)
        if i < num_steps - 1:
            denoised = net(x_next, t_next_in, y).to(torch.float32)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            print(f'Sample timestep {num_steps-i}', end='\r')
            x_next = sample_step(net, num_steps, i, t_cur, t_next, x_next, y)

    return x_next


def sample_ddp(net, num_samples, dim, num_steps=50, y=None, rank=None):
    """
    `sample` function modified for `net` that are wrapped in torch.nn.parallel.DistributedDataParallel.

    For the rest, see `sample`.
    """
    latents = torch.randn([num_samples, dim]).to(rank)

    step_indices = torch.arange(num_steps, dtype=torch.float32).to(rank)

    sigma_min = max(SIGMA_MIN, net.module.sigma_min)
    sigma_max = min(SIGMA_MAX, net.module.sigma_max)

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.module.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float32) * t_steps[0]
    y = y.to(rank)

    def sample_step(net, num_steps, i, t_cur, t_next, x_next, y, rank):

        x_cur = x_next
        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.module.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        # Euler step.

        t_hat_in = torch.full((x_hat.shape[0], 1), t_hat).to(rank)
        denoised = net(x_hat, t_hat_in, y).to(torch.float32)

        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            t_next_in = torch.full((x_hat.shape[0], 1), t_next).to(rank)
            denoised = net(x_next, t_next_in, y).to(torch.float32)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            print(f'Sample timestep {num_steps-i:4d}', end='\r')
            x_next = sample_step(net, num_steps, i, t_cur, t_next, x_next, y, rank)

    return x_next


class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, denosie_fn, data, labels, augment_pipe=None):
        rnd_uniform = torch.rand([data.shape[0], 1, 1, 1], device=data.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
        n = torch.randn_like(y) * sigma
        D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100, D=128, N=3072, opts=None):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.D = D
        self.N = N
        print(f"In VE loss: D:{self.D}, N:{self.N}")

    def __call__(self, denosie_fn, data, labels = None, augment_pipe=None, stf=False, pfgmpp=False, ref_data=None):
        if pfgmpp:

            # N, 
            rnd_uniform = torch.rand(data.shape[0], device=data.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)

            r = sigma.double() * np.sqrt(self.D).astype(np.float64)
            # Sampling form inverse-beta distribution
            samples_norm = np.random.beta(a=self.N / 2., b=self.D / 2.,
                                          size=data.shape[0]).astype(np.double)

            samples_norm = np.clip(samples_norm, 1e-3, 1-1e-3)

            inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
            inverse_beta = torch.from_numpy(inverse_beta).to(data.device).double()
            # Sampling from p_r(R) by change-of-variable
            samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
            samples_norm = samples_norm.view(len(samples_norm), -1)
            # Uniformly sample the angle direction
            gaussian = torch.randn(data.shape[0], self.N).to(samples_norm.device)
            unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
            # Construct the perturbation for x
            perturbation_x = unit_gaussian * samples_norm
            perturbation_x = perturbation_x.float()

            sigma = sigma.reshape((len(sigma), 1, 1, 1))
            weight = 1 / sigma ** 2
            y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
            n = perturbation_x.view_as(y)
            D_yn = denosie_fn(y + n, sigma, labels,  augment_labels=augment_labels)
        else:
            rnd_uniform = torch.rand([data.shape[0], 1, 1, 1], device=data.device)
            sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
            weight = 1 / sigma ** 2
            y, augment_labels = augment_pipe(data) if augment_pipe is not None else (data, None)
            n = torch.randn_like(y) * sigma
            D_yn = denosie_fn(y + n, sigma, labels, augment_labels=augment_labels)

        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, hid_dim = 100, gamma=5, opts=None):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.hid_dim = hid_dim
        self.gamma = gamma
        self.opts = opts


    def __call__(self, denoise_fn, data, y):

        rnd_normal = torch.randn(data.shape[0], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # y = data
        n = torch.randn_like(data) * sigma.unsqueeze(1)
        D_yn = denoise_fn(data + n, sigma, y)
    
        target = data
        loss = weight.unsqueeze(1) * ((D_yn - target) ** 2)

        return loss


#--------------------------Multimodal tabsyn---------------------------
class Joint_net(nn.Module):
    """
    Joint neural network for multimodal tabsyn.
    
    Input the concatenated latent embeddings. Then split the input
    and provide to the MLP network. Concatenate the output and return
    it. Suitable for sampling functions.

    Args:
        denoise_fn: shall take input x0, x1, t0, t1 and output x0, x1.
    """
    def __init__(self, denoise_fn, d_in0, d_in1, device=None):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.d_in0 = d_in0
        self.d_in1 = d_in1
        self.device = device
        
    def forward(self, x, timesteps):
        """
        Args:
            x: the concatenated input
            timesteps: the timesteps. 
        """
        x0 = x[:,:self.d_in0]
        x1 = x[:,self.d_in0:]
        timesteps = timesteps.to(torch.float32).reshape(-1, 1)
        timesteps = timesteps.flatten()
        x0_out, x1_out = self.denoise_fn(x0, x1, timesteps, timesteps)
        x_out = torch.cat((x0_out, x1_out), dim=1)

        return x_out


class Dat2y_net(nn.Module):
    """
    Multimodal tabsyn neural network for data to y conditional generation.
    
    Input the concatenated latent embeddings. Then split the input
    and provide to the MLP network. Concatenate the output and return
    it. Suitable for sampling functions.
    """
    def __init__(self, denoise_fn, dat, device=None):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.dat = dat
        self.device = device
        
    def forward(self, y, timesteps):
        """
        Args:
            y: the input, (being denoised) y.
            timesteps: the timesteps. 
            dat: the real data to condition on.
        """
        t_y = timesteps.to(torch.float32).reshape(-1, 1).flatten()
        t_dat =  torch.zeros(timesteps.size(0), dtype=torch.int, device=self.device)
        
        x0_out, x1_out = self.denoise_fn(self.dat, y, t_dat, t_y)
        # x_out = torch.cat((x0_out, x1_out), dim=1)
        return x1_out


def _round_sigma(sigma):
    """
    Helper function for `sample_multimodal`.
    """
    return torch.as_tensor(sigma)


def sample_multimodal(net, num_samples, dim, num_steps=50, device=None):
    """
    Sampling function for uni-tabsyn.
    
    Args:
        net: x0 prediction network
        dim: the dimension for the generated data, which will be input to `net`.
    """
    latents = torch.randn([num_samples, dim], device=device)

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)

    sigma_min = SIGMA_MIN
    sigma_max = SIGMA_MAX

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([_round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    x_next = latents.to(torch.float32) * t_steps[0]

    def sample_step(net, num_steps, i, t_cur, t_next, x_next):

        x_cur = x_next
        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = _round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
        
        # Euler step.
        t_hat_in = torch.full((x_hat.shape[0], 1), t_hat).to(x_hat.device)
        denoised = net(x_hat, t_hat_in).to(torch.float32)

        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            t_next_in = torch.full((x_hat.shape[0], 1), t_next).to(x_hat.device)
            denoised = net(x_next, t_next_in).to(torch.float32)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            print(f'Sample timestep {num_steps-i:4d}', end='\r')
            x_next = sample_step(net, num_steps, i, t_cur, t_next, x_next)

    return x_next
