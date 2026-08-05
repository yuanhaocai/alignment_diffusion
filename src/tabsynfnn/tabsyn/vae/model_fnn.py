import math

# Portions adapted from amazon-science/TabSyn (Apache-2.0) and modified.
import torch
import torch.nn as nn
import torch.nn.init as nn_init
from torch import Tensor


class Tokenizer(nn.Module):
    def __init__(self, d_numerical, categories, d_token, bias):
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'The tokenizer category_embeddings weight has shape {self.category_embeddings.weight.shape}')

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self):
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num, x_cat):
        """
        Args:
            x_num: could be of shape (batch_size, 0) to indicate the non-existence of num data.
            x_cat: could be `None` to indicate the non-existence.
        """
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            # + ([] if x_num is None else [x_num]),
            + ([] if x_num.shape[1] == 0 else [x_num]),
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]

        if x_cat is not None:
            if x_cat.shape[1] != 0:
                x = torch.cat(
                    [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                    dim=1,
                )

        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]

        return x


class Reconstructor(nn.Module):
    """
    Able to deal with the case when there is only numerical data in the latent
    embeddings.
    """
    def __init__(self, d_numerical, categories, d_token):
        super(Reconstructor, self).__init__()

        self.d_numerical = d_numerical
        self.weight = nn.Parameter(Tensor(d_numerical, d_token))
        nn.init.xavier_uniform_(self.weight, gain=1 / math.sqrt(2))
        self.cat_recons = nn.ModuleList()

        if categories is not None:
            for d in categories:
                recon = nn.Linear(d_token, d)
                nn.init.xavier_uniform_(recon.weight, gain=1 / math.sqrt(2))
                self.cat_recons.append(recon)

    def forward(self, h):
        """
        Args:
            h: be of shape (n, seq_len, d_token).
        """
        h_num = h[:, :self.d_numerical]
        h_cat = h[:, self.d_numerical:]

        recon_x_num = torch.mul(h_num, self.weight.unsqueeze(0)).sum(-1) if self.d_numerical != 0 else torch.ones(h_num.shape[0], 0).to(h_num.device)
        recon_x_cat = []

        for i, recon in enumerate(self.cat_recons):
            recon_x_cat.append(recon(h_cat[:, i]))

        return recon_x_num, recon_x_cat


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hid_dims):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hid_dims)
        self.fc_mu = nn.Linear(hid_dims, latent_dim)
        self.fc_logvar = nn.Linear(hid_dims, latent_dim)
    
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, output_dim, latent_dim, hid_dims):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hid_dims)
        self.fc2 = nn.Linear(hid_dims, output_dim)
        self.final = nn.Linear(output_dim, output_dim)
    
    def forward(self, z):
        h1 = torch.relu(self.fc1(z))
        h2 = torch.tanh(self.fc2(h1))
        return self.final(h2)


class VAE(nn.Module):
    def __init__(self, d_numerical, categories, d_token, seq_len, latent_dim, bias=True):
        """
        Args:
            latent_dim: the final dimension of the latent space.
        """
        super(VAE, self).__init__()
 
        self.vae_in_dim = (seq_len + 1) * d_token
        self.vae_out_dim = seq_len * d_token
        self.hid_dims = math.ceil(seq_len * d_token / 2)
        self.latent_dim = latent_dim
 
        self.Tokenizer = Tokenizer(d_numerical, categories, d_token, bias=bias)
        self.encoder = Encoder(self.vae_in_dim, latent_dim, self.hid_dims)
        self.decoder = Decoder(self.vae_out_dim, latent_dim, self.hid_dims)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sample(self, num_samples: int, device) -> Tensor:
        """
        Samples from the latent space and go through the decoder.
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decoder(z)
        return samples

    def forward(self, x_num, x_cat):
        """
        h: of shape (batch_size, self.vae_out_dim)
        """
        x = self.Tokenizer(x_num, x_cat)
        mu_z, std_z = self.encoder(x.view(x.shape[0], -1))
        z = self.reparameterize(mu_z, std_z)
        h = self.decoder(z)
        return h, mu_z, std_z


class Model_VAE_fnn(nn.Module):
    def __init__(self, d_numerical, categories, d_token, seq_len, latent_dim, bias = True):
        super(Model_VAE_fnn, self).__init__()
        self.d_token = d_token
        self.seq_len = seq_len
        self.VAE = VAE(d_numerical, categories, d_token, seq_len, latent_dim, bias = bias)
        self.Reconstructor = Reconstructor(d_numerical, categories, d_token)

    def sample(self, num_samples: int, device: int, multigpu: bool, device_ids: list = None) -> Tensor:
        """
        Samples from the latent space and go through the decoder.
        Args:
            num_samples: (Int) Number of samples
        
        Return:
            a tuple, (np.array, list of np.arrays)
        """
        if multigpu:
            self.VAE.decoder = nn.DataParallel(self.VAE.decoder, device_ids=device_ids)
        
        z = self.VAE.sample(num_samples, device)
        recon_x_num, recon_x_cat = self.Reconstructor(z.view(z.shape[0], self.seq_len, self.d_token))
        return recon_x_num, recon_x_cat

    def forward(self, x_num, x_cat):
        h, mu_z, std_z = self.VAE(x_num, x_cat)
        recon_x_num, recon_x_cat = self.Reconstructor(h.view(h.shape[0], self.seq_len, self.d_token))
        return recon_x_num, recon_x_cat, mu_z, std_z


class Encoder_model_fnn(nn.Module):
    def __init__(self, d_numerical, categories, d_token, seq_len, latent_dim, bias=True):
        super(Encoder_model_fnn, self).__init__()
        self.vae_in_dim = (seq_len + 1) * d_token
        self.hid_dims = math.ceil(seq_len * d_token / 2)
        self.Tokenizer = Tokenizer(d_numerical, categories, d_token, bias)
        self.VAE_Encoder = Encoder(self.vae_in_dim, latent_dim, self.hid_dims)

    def load_weights(self, Pretrained_VAE):
        self.Tokenizer.load_state_dict(Pretrained_VAE.VAE.Tokenizer.state_dict())
        self.VAE_Encoder.load_state_dict(Pretrained_VAE.VAE.encoder.state_dict())

    def forward(self, x_num, x_cat):
        x = self.Tokenizer(x_num, x_cat)
        z, std = self.VAE_Encoder(x.view(x.shape[0], -1))
        return z


class Decoder_model_fnn(nn.Module):
    def __init__(self, d_numerical, categories, d_token, seq_len, latent_dim, bias=True):
        super(Decoder_model_fnn, self).__init__()
        self.seq_len = seq_len
        self.d_token = d_token
        self.vae_out_dim = seq_len * d_token
        self.hid_dims = math.ceil(seq_len * d_token / 2)
        self.VAE_Decoder = Decoder(self.vae_out_dim, latent_dim, self.hid_dims)
        self.Detokenizer = Reconstructor(d_numerical, categories, d_token)
        
    def load_weights(self, Pretrained_VAE):
        self.VAE_Decoder.load_state_dict(Pretrained_VAE.VAE.decoder.state_dict())
        self.Detokenizer.load_state_dict(Pretrained_VAE.Reconstructor.state_dict())

    def forward(self, z):
        h = self.VAE_Decoder(z)
        x_hat_num, x_hat_cat = self.Detokenizer(h.view(h.shape[0], self.seq_len, self.d_token))
        return x_hat_num, x_hat_cat
