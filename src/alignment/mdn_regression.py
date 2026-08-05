"""
Mixture Density Network (MDN) utilities for univariate regression.

An MDN models the full conditional density p(y | x), not just a point estimate.
For each input row x, the network predicts a Gaussian mixture:

    p(y | x) = sum_k pi_k(x) Normal(y; mu_k(x), sigma_k(x)^2)

where pi_k are mixture weights, mu_k are component means, and sigma_k are
positive component standard deviations. The point prediction is the mixture
mean, sum_k pi_k * mu_k. Prediction intervals are computed from central
quantiles of the learned mixture distribution.

Typical usage:

    mdn = MixtureDensityNetworkRegressor(
        n_components=5,
        hidden_dims=(128, 64),
        max_epochs=200,
        batch_size=512,
        lr=1e-3,
        validation_fraction=0.2,
        patience=20,
        random_state=0,
        verbose=1,
        log_every=10,
    )
    mdn.fit(X_train, y_train)
    pred = mdn.predict(X_test)
    weights, mus, sigmas = mdn.predict_dist(X_test)
    intervals = mdn.predict_interval(X_test, alpha=0.05)
    history = mdn.history_
"""

import copy

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def _mps_is_available():
    mps_backend = getattr(torch.backends, "mps", None)
    return mps_backend is not None and mps_backend.is_available()


def _set_torch_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if _mps_is_available() and hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)


class _GaussianMDN(nn.Module):
    def __init__(self, input_dim, n_components, hidden_dims, min_sigma):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        self.feature_net = nn.Sequential(*layers)
        self.param_layer = nn.Linear(prev_dim, 3 * n_components)
        self.min_sigma = min_sigma

    def forward(self, x):
        params = self.param_layer(self.feature_net(x))
        logits, mus, raw_sigmas = torch.chunk(params, chunks=3, dim=1)
        sigmas = F.softplus(raw_sigmas) + self.min_sigma
        return logits, mus, sigmas


def gaussian_mixture_quantile(weights, mus, sigmas, q, max_iter=80):
    weights = np.asarray(weights, dtype=float)
    mus = np.asarray(mus, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    row_sums = weights.sum(axis=1, keepdims=True)
    weights = weights / row_sums

    lower = np.min(mus - 10 * sigmas, axis=1)
    upper = np.max(mus + 10 * sigmas, axis=1)
    for _ in range(max_iter):
        mid = 0.5 * (lower + upper)
        cdf = np.sum(weights * stats.norm.cdf((mid[:, None] - mus) / sigmas), axis=1)
        lower = np.where(cdf < q, mid, lower)
        upper = np.where(cdf >= q, mid, upper)
    return 0.5 * (lower + upper)


def gaussian_mixture_interval(weights, mus, sigmas, alpha=0.05):
    lower = gaussian_mixture_quantile(weights, mus, sigmas, alpha / 2)
    upper = gaussian_mixture_quantile(weights, mus, sigmas, 1 - alpha / 2)
    return np.column_stack([lower, upper])


class MixtureDensityNetworkRegressor:
    """
    Sklearn-style PyTorch MDN regressor for one-dimensional continuous targets.

    Model details
    -------------
    The model is a feed-forward neural network with ReLU hidden layers. Its final
    layer emits three vectors per observation:

    - ``logits``: converted with softmax into mixture weights ``pi``.
    - ``mus``: Gaussian component means.
    - ``raw_sigmas``: converted with softplus into positive standard deviations.

    Training minimizes the negative log likelihood of the observed target under
    the predicted Gaussian mixture:

        -log sum_k pi_k(x_i) Normal(y_i; mu_k(x_i), sigma_k(x_i)^2)

    Public methods
    --------------
    - ``fit(X, y)``: trains the MDN.
    - ``predict(X)``: returns the predictive mean, ``sum_k pi_k * mu_k``.
    - ``predict_dist(X)``: returns ``weights, mus, sigmas`` arrays, each with
      shape ``(n_observations, n_components)``.
    - ``predict_interval(X, alpha=0.05)``: returns central prediction intervals
      with shape ``(n_observations, 2)`` using mixture quantiles.
    - ``nll(X, y)``: returns average negative log likelihood for any dataset.

    Hyperparameters
    ---------------
    n_components:
        Number of Gaussian mixture components. More components can represent
        multimodal or skewed conditional densities, but can also overfit or
        collapse if the dataset is small.
    hidden_dims:
        Widths of the hidden ReLU layers. For example, ``(128, 64)`` creates two
        hidden layers.
    lr:
        AdamW learning rate.
    weight_decay:
        AdamW L2-style regularization.
    batch_size:
        Mini-batch size for training and prediction.
    max_epochs:
        Maximum number of training epochs.
    validation_fraction:
        Fraction of training data held out for early stopping. Set to ``0`` or
        ``None`` to monitor training loss instead.
    patience:
        Number of epochs without validation-loss improvement before stopping.
    min_delta:
        Minimum monitored-loss improvement required to reset early stopping.
    min_sigma:
        Lower offset added after softplus so predicted standard deviations stay
        strictly positive.
    grad_clip:
        Maximum gradient norm. Set to ``None`` to disable gradient clipping.
    random_state:
        Seed for NumPy, PyTorch, and the train/validation split.
    device:
        Optional explicit PyTorch device string, e.g. ``"cpu"``, ``"cuda"``, or
        ``"mps"``. If omitted, the regressor chooses CUDA, then MPS, then CPU.
    verbose:
        If positive, print epoch progress during ``fit``.
    log_every:
        Print every ``log_every`` epochs when ``verbose`` is positive. The first
        epoch and the final/early-stopping epoch are also printed. Set
        ``verbose=2`` to additionally print every improved epoch.

    Notes
    -----
    If the caller standardizes the regression target, predictions and intervals
    remain on that standardized scale unless the caller inverse-transforms them.
    """

    def __init__(
        self,
        n_components=5,
        hidden_dims=(128, 64),
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=512,
        max_epochs=200,
        validation_fraction=0.2,
        patience=20,
        min_delta=1e-4,
        min_sigma=1e-3,
        grad_clip=5.0,
        random_state=0,
        device=None,
        verbose=0,
        log_every=10,
    ):
        self.n_components = n_components
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.min_delta = min_delta
        self.min_sigma = min_sigma
        self.grad_clip = grad_clip
        self.random_state = random_state
        self.device = device
        self.verbose = verbose
        self.log_every = log_every

    def _get_device(self):
        if self.device is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _nll(logits, mus, sigmas, y):
        log_weights = F.log_softmax(logits, dim=1)
        log_probs = torch.distributions.Normal(mus, sigmas).log_prob(y[:, None])
        return -torch.logsumexp(log_weights + log_probs, dim=1).mean()

    def fit(self, X, y, X_val=None, y_val=None, verbose=None, log_every=None):
        """
        Fit the MDN and store per-epoch training history in ``history_``.

        ``X_val`` and ``y_val`` can be supplied to use an explicit validation
        set for early stopping and logging. If they are omitted,
        ``validation_fraction`` controls whether a validation split is carved
        out of ``X`` and ``y``.
        """
        _set_torch_seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        verbose = self.verbose if verbose is None else verbose
        log_every = self.log_every if log_every is None else log_every

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of observations.")

        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must be supplied together.")

        if X_val is not None:
            X_t, y_t = X, y
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32).reshape(-1)
            if X_val.ndim != 2:
                raise ValueError("X_val must be a 2D array.")
            if X_val.shape[0] != y_val.shape[0]:
                raise ValueError("X_val and y_val must contain the same number of observations.")
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
            )
        elif self.validation_fraction and X.shape[0] >= 10:
            X_t, X_val, y_t, y_val = train_test_split(
                X, y, test_size=self.validation_fraction, random_state=self.random_state
            )
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
            )
        else:
            X_t, y_t = X, y
            val_dataset = None

        self.device_ = self._get_device()
        self.model_ = _GaussianMDN(X.shape[1], self.n_components, self.hidden_dims, self.min_sigma)
        self.model_.to(self.device_)

        train_dataset = TensorDataset(
            torch.tensor(X_t, dtype=torch.float32),
            torch.tensor(y_t, dtype=torch.float32),
        )
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=True,
            generator=generator,
        )
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=min(self.batch_size, len(val_dataset)),
                shuffle=False,
            )

        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        best_loss = np.inf
        best_state = copy.deepcopy(self.model_.state_dict())
        best_epoch = 0
        epochs_without_improvement = 0
        self.history_ = []

        if verbose:
            print(
                "Training MDN "
                f"(device={self.device_}, n_components={self.n_components}, "
                f"hidden_dims={self.hidden_dims}, train_size={len(train_dataset)}, "
                f"val_size={0 if val_dataset is None else len(val_dataset)})"
            )

        for epoch in range(self.max_epochs):
            self.model_.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device_)
                batch_y = batch_y.to(self.device_)

                optimizer.zero_grad()
                logits, mus, sigmas = self.model_(batch_X)
                loss = self._nll(logits, mus, sigmas, batch_y)
                loss.backward()
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model_.parameters(), self.grad_clip)
                optimizer.step()

                train_loss += loss.item() * batch_X.shape[0]

            train_loss /= len(train_dataset)
            if val_loader is None:
                val_loss = None
                monitor_loss = train_loss
            else:
                self.model_.eval()
                val_loss_total = 0.0
                with torch.no_grad():
                    for val_X, val_y in val_loader:
                        val_X = val_X.to(self.device_)
                        val_y = val_y.to(self.device_)
                        logits, mus, sigmas = self.model_(val_X)
                        loss = self._nll(logits, mus, sigmas, val_y)
                        val_loss_total += loss.item() * val_X.shape[0]
                    val_loss = val_loss_total / len(val_dataset)
                    monitor_loss = val_loss

            improved = False
            if monitor_loss < best_loss - self.min_delta:
                improved = True
                best_loss = monitor_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                best_epoch = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            history_row = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": None if val_loss is None else float(val_loss),
                "monitor_loss": float(monitor_loss),
                "best_loss": float(best_loss),
                "is_best": improved,
            }
            self.history_.append(history_row)

            should_log = (
                bool(verbose)
                and (
                    epoch == 0
                    or (verbose >= 2 and improved)
                    or (log_every and (epoch + 1) % log_every == 0)
                    or epochs_without_improvement >= self.patience
                    or epoch + 1 == self.max_epochs
                )
            )
            if should_log:
                val_text = "NA" if val_loss is None else f"{val_loss:.6f}"
                print(
                    f"epoch {epoch + 1:04d} | train_nll={train_loss:.6f} | "
                    f"val_nll={val_text} | best={best_loss:.6f} | "
                    f"patience={epochs_without_improvement}/{self.patience}"
                )

            if epochs_without_improvement >= self.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}; best epoch was {best_epoch}.")
                break

        self.model_.load_state_dict(best_state)
        self.n_epochs_ = epoch + 1
        self.best_loss_ = best_loss
        self.best_epoch_ = best_epoch
        return self

    def nll(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        total_loss = 0.0
        self.model_.eval()
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device_)
                batch_y = batch_y.to(self.device_)
                logits, mus, sigmas = self.model_(batch_X)
                loss = self._nll(logits, mus, sigmas, batch_y)
                total_loss += loss.item() * batch_X.shape[0]
        return total_loss / len(dataset)

    def predict_dist(self, X):
        X = np.asarray(X, dtype=np.float32)
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        weights_all, mus_all, sigmas_all = [], [], []
        self.model_.eval()
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(self.device_)
                logits, mus, sigmas = self.model_(batch_X)
                weights_all.append(F.softmax(logits, dim=1).cpu().numpy())
                mus_all.append(mus.cpu().numpy())
                sigmas_all.append(sigmas.cpu().numpy())

        return np.vstack(weights_all), np.vstack(mus_all), np.vstack(sigmas_all)

    def predict(self, X):
        weights, mus, _ = self.predict_dist(X)
        return np.sum(weights * mus, axis=1)

    def predict_interval(self, X, alpha=0.05):
        weights, mus, sigmas = self.predict_dist(X)
        return gaussian_mixture_interval(weights, mus, sigmas, alpha=alpha)
