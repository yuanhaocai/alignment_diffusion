import math

# Portions adapted from amazon-science/TabSyn (Apache-2.0) and modified.
import torch
import torch.nn as nn


#----------------------------------Tokenzier for categorical y--------------------------------
class Tokenizer_caty(nn.Module):
    """
    Tokenzier class for categorical y or all categorical features.
    y can be multi-dimensional.
    """
    def __init__(self, categories, d_token, bias=True):
        super().__init__()
        d_bias = len(categories)
        category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
        self.register_buffer('category_offsets', category_offsets)
        self.category_embeddings = nn.Embedding(sum(categories), d_token)
        nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.Tensor(d_bias, d_token))
        nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, y):
        """
        Args:
            y: shall be categorical.

        Output:
            Of shape (batch_size, y_dim/cat_dim, d_token)
        """
        y_ = self.category_embeddings(y + self.category_offsets[None])
        y_ = y_ + self.bias[None]

        return y_
    

class Reconstructor_caty(nn.Module):
    """
    Reconstructor class for categorical y or all categorical features.

    Reconstructs the latent embeddings to arrays with same dimension as one-hot encoded ones. 
    Will then need to use argmax to get the predicted level.
    """
    def __init__(self, categories, d_token):
        super().__init__()
        self.categories = categories
        self.cat_recons = nn.ModuleList()

        for d in categories:
            recon = nn.Linear(d_token, d)
            nn.init.xavier_uniform_(recon.weight, gain=1/math.sqrt(2))
            self.cat_recons.append(recon)

    def forward(self, h):
        """
        Args:
            h: shall be of dim (batch_size, y_dim/cat_dim, d_token).

        Output:
            Reconstructed y of shape same as one-hot encoded array, but not containing 0s and 1s,
            but any values to use argmax with.
        """
        recon_caty = []
        for i, recon in enumerate(self.cat_recons):
            recon_caty.append(recon(h[:, i]))

        return recon_caty


class VAE_caty(nn.Module):
    """
    VAE class for encoding categorical y. Currently only work for 1-dimensional y.
    """
    def __init__(self, categories, d_token, bias=True):
        super().__init__()
        self.d_token = d_token
        self.tokenizer = Tokenizer_caty(categories, d_token, bias=bias)
        self.reconstructor = Reconstructor_caty(categories, d_token)

    def forward(self, y):
        y_embeddings = self.tokenizer(y)
        recon_caty = self.reconstructor(y_embeddings)
        return recon_caty, y_embeddings


class Encoder_caty(nn.Module):
    def __init__(self, categories, d_token, bias=True):
        super().__init__()
        self.tokenizer = Tokenizer_caty(categories, d_token, bias)

    def load_weights(self, Pretrained_VAE):
        self.tokenizer.load_state_dict(Pretrained_VAE.tokenizer.state_dict())

    def forward(self, y):
        return self.tokenizer(y)


class Decoder_caty(nn.Module):
    def __init__(self, categories, d_token):
        super().__init__()
        self.d_token = d_token
        self.detokenizer = Reconstructor_caty(categories, d_token)
        
    def load_weights(self, Pretrained_VAE):
        self.detokenizer.load_state_dict(Pretrained_VAE.reconstructor.state_dict())

    def forward(self, y_embeddings):
        return self.detokenizer(y_embeddings)
    



# class Tokenizer_cat(nn.Module):
#     """
#     The Tokenizer for categorical features, for use in cross-attention. So `d_token` shall be set to 
#     just 1.
#     """
#     def __init__(self, categories, d_token=1, bias=True):
#         super().__init__()
#         d_bias = len(categories)
#         category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
#         self.register_buffer('category_offsets', category_offsets)
#         self.category_embeddings = nn.Embedding(sum(categories), d_token)
#         nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

#         self.bias = nn.Parameter(torch.Tensor(d_bias, d_token)) if bias else None
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))

#     def forward(self, x_cat):
#         """
#         Args:
#             x_cat: of shape (batch_size, cat_dim).
        
#         Return:
#             x: of shape (batch_size, cat_dim, d_token).
#         """
#         x = self.category_embeddings(x_cat + self.category_offsets[None])
#         if self.bias is not None:
#             x = x + self.bias[None]
#         return x


# class Reconstructor_cat(nn.Module):
#     """
#     Reconstructor for categorical features.
#     """
#     def __init__(self, categories, d_token):
#         super().__init__()
#         self.cat_recons = nn.ModuleList()

#         for d in categories:
#             recon = nn.Linear(d_token, d)
#             nn.init.xavier_uniform_(recon.weight, gain=1 / math.sqrt(2))
#             self.cat_recons.append(recon)

#     def forward(self, h):
#         """
#         Args:
#             h: be of shape (n, seq_len, d_token).
#         """
#         h_cat = h
#         recon_x_cat = []
#         for i, recon in enumerate(self.cat_recons):
#             recon_x_cat.append(recon(h_cat[:, i]))

#         return recon_x_cat
