# Portions adapted from amazon-science/TabSyn (Apache-2.0) and modified.
"""
Uncertainty quantification utils, including both prediction interval (for regression)
and multiclass versin of calibration error and ECE (for classification)
"""
import os
import argparse
from typing import Literal

import numpy as np
import scipy.stats as stats
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error.

    Binary:
        y_prob has shape (n_samples,) with P(y=1).

    Multiclass:
        y_prob has shape (n_samples, n_classes). This computes top-label ECE:
        bin by confidence max_k P(y=k), and compare bin confidence with
        bin accuracy mean(argmax_k P(y=k) == y_true).
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob)

    if y_prob.ndim == 1:
        confidences = y_prob
        accuracies = (y_true == 1).astype(float)
    elif y_prob.ndim == 2:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true).astype(float)
    else:
        raise ValueError("y_prob must be a 1D binary probability vector or a 2D multiclass probability matrix.")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.clip(np.digitize(confidences, bins, right=True) - 1, 0, n_bins - 1)

    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if np.any(mask):
            conf_mean = np.mean(confidences[mask])
            acc_mean = np.mean(accuracies[mask])
            ece += (np.sum(mask) / len(confidences)) * np.abs(conf_mean - acc_mean)
    return ece


def multiclass_brier_score(y_true, y_prob, labels=None):
    """
    Multiclass Brier score using one-hot targets:
        mean_i sum_k (p_ik - 1{y_i=k})^2
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob)

    if y_prob.ndim != 2:
        raise ValueError("For multiclass Brier score, y_prob must have shape (n_samples, n_classes).")

    if labels is None:
        labels = np.arange(y_prob.shape[1])
    labels = np.asarray(labels)

    if len(labels) != y_prob.shape[1]:
        raise ValueError("labels must have the same length as y_prob.shape[1].")

    label_to_index = {label: idx for idx, label in enumerate(labels)}
    y_indices = np.array([label_to_index[y] for y in y_true])
    y_onehot = np.zeros_like(y_prob, dtype=float)
    y_onehot[np.arange(len(y_true)), y_indices] = 1.0

    return np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1))


def coverage_and_width(y_true, lower_bound, upper_bound):
    coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
    width = np.mean(upper_bound - lower_bound)
    return coverage, width

def crps_gaussian(y_true, mu, sigma):
    z = (y_true - mu) / sigma
    pdf = stats.norm.pdf(z)
    cdf = stats.norm.cdf(z)
    crps = sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
    return np.mean(crps)


def _normal_absolute_moment(delta, sigma):
    sigma = np.maximum(np.asarray(sigma, dtype=float), np.finfo(float).eps)
    delta = np.asarray(delta, dtype=float)
    z = delta / sigma
    return 2 * sigma * stats.norm.pdf(z) + delta * (2 * stats.norm.cdf(z) - 1)


def crps_gaussian_mixture(y_true, weights, mus, sigmas, return_pointwise=False, batch_size=4096):
    """
    Closed-form CRPS for a univariate Gaussian mixture predictive distribution.

    For F = sum_k w_k N(mu_k, sigma_k^2), this uses
        CRPS(F, y) = E|X - y| - 0.5 E|X - X'|
    where X and X' are independent draws from F.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float)
    mus = np.asarray(mus, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)

    if weights.ndim != 2 or mus.shape != weights.shape or sigmas.shape != weights.shape:
        raise ValueError("weights, mus, and sigmas must all have shape (n_observations, n_components).")
    if weights.shape[0] != y_true.size:
        raise ValueError("weights, mus, and sigmas must have one row per y_true value.")
    if np.any(sigmas <= 0):
        raise ValueError("All Gaussian mixture sigmas must be positive.")

    row_sums = weights.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Each Gaussian mixture row must have positive total weight.")
    weights = weights / row_sums

    crps_values = []
    for start in range(0, y_true.size, batch_size):
        stop = min(start + batch_size, y_true.size)
        y_batch = y_true[start:stop]
        w_batch = weights[start:stop]
        mu_batch = mus[start:stop]
        sigma_batch = sigmas[start:stop]

        term_1 = np.sum(
            w_batch * _normal_absolute_moment(y_batch[:, None] - mu_batch, sigma_batch),
            axis=1,
        )

        mu_delta = mu_batch[:, :, None] - mu_batch[:, None, :]
        sigma_pair = np.sqrt(sigma_batch[:, :, None] ** 2 + sigma_batch[:, None, :] ** 2)
        weight_pair = w_batch[:, :, None] * w_batch[:, None, :]
        term_2 = np.sum(weight_pair * _normal_absolute_moment(mu_delta, sigma_pair), axis=(1, 2))

        crps_values.append(term_1 - 0.5 * term_2)

    crps_values = np.concatenate(crps_values)
    if return_pointwise:
        return crps_values
    return np.mean(crps_values)


def crps_gaussian_from_intervals(y_true, mu, intervals, alpha=0.1):
    """
    Gaussian-approximation CRPS from central prediction intervals.

    This treats the interval width as if it came from a Gaussian central
    ``1 - alpha`` interval around ``mu``.
    """
    y_true = np.asarray(y_true).reshape(-1)
    mu = np.asarray(mu).reshape(-1)
    intervals = np.asarray(intervals, dtype=float)

    if intervals.shape != (y_true.size, 2):
        raise ValueError("intervals must have shape (n_observations, 2).")
    if mu.size != y_true.size:
        raise ValueError("mu must have the same length as y_true.")

    lower = intervals[:, 0]
    upper = intervals[:, 1]
    z = stats.norm.ppf(1 - alpha / 2)
    sigma = (upper - lower) / (2 * z)
    sigma = np.maximum(sigma, np.finfo(float).eps)
    return crps_gaussian(y_true, mu, sigma)


def crps_empirical(y_true, samples, sample_axis=0, return_pointwise=False):
    """
    Empirical CRPS for Monte Carlo samples from a predictive distribution.

    For each observation, this computes
        E|X - y| - 0.5 E|X - X'|
    using the empirical distribution of the supplied samples. ``samples`` can
    be one array or a list/tuple of arrays, in which case the arrays are treated
    as one equally weighted sample pool.
    """
    y_true = np.asarray(y_true).reshape(-1)
    if isinstance(samples, tuple):
        sample_sets = samples
    elif isinstance(samples, list) and samples and all(np.asarray(sample_set).ndim >= 2 for sample_set in samples):
        sample_sets = samples
    else:
        sample_sets = (samples,)
    sample_sets = [np.asarray(sample_set) for sample_set in sample_sets]

    normalized_sets = []
    for sample_set in sample_sets:
        if sample_set.ndim == 1:
            if y_true.size != 1:
                raise ValueError("1D samples are only valid for a single observation.")
            sample_set = sample_set.reshape(-1, 1)
        elif sample_set.ndim == 2:
            sample_set = np.moveaxis(sample_set, sample_axis, 0)
        else:
            raise ValueError("Each sample array must be 1D or 2D.")

        if sample_set.shape[1] != y_true.size:
            raise ValueError(
                "Each sample array must have one observation column per y_true value "
                "after moving sample_axis to axis 0."
            )
        normalized_sets.append(sample_set)

    crps_values = []
    weight_cache = {}
    for obs_idx, y in enumerate(y_true):
        obs_samples = [sample_set[:, obs_idx].reshape(-1) for sample_set in normalized_sets]
        if len(obs_samples) == 1:
            ensemble = obs_samples[0]
        else:
            ensemble = np.concatenate(obs_samples)

        ensemble = np.sort(ensemble.astype(float, copy=False))
        n_samples = ensemble.size
        if n_samples == 0:
            raise ValueError("At least one Monte Carlo sample is required.")

        mean_abs_error = np.mean(np.abs(ensemble - y))
        weights = weight_cache.get(n_samples)
        if weights is None:
            weights = 2 * np.arange(1, n_samples + 1) - n_samples - 1
            weight_cache[n_samples] = weights
        ensemble_spread = np.sum(weights * ensemble) / (n_samples ** 2)
        crps_values.append(mean_abs_error - ensemble_spread)

    crps_values = np.asarray(crps_values)
    if return_pointwise:
        return crps_values
    return np.mean(crps_values)


def _as_1d_labels(y):
    return np.asarray(y).reshape(-1)


def labels_to_indices(y, class_labels):
    label_to_index = {class_label: index for index, class_label in enumerate(class_labels)}
    try:
        return np.asarray([label_to_index[value] for value in _as_1d_labels(y)])
    except KeyError as exc:
        raise ValueError(f"Found label not present in class_labels: {exc.args[0]}") from exc


def infer_class_labels(y_true, y_num_classes):
    """
    Infer class labels in the same order used by scikit-learn predict_proba outputs.

    For ordinal labels encoded as 0..K-1, this keeps that natural order while preserving
    the loaded label dtype, e.g. string labels stay as ["0", "1", ...].
    """
    y_true = _as_1d_labels(y_true)
    if y_num_classes is None:
        return np.unique(y_true)

    numeric_labels = np.arange(y_num_classes)
    string_labels = numeric_labels.astype(str)
    unique_as_strings = set(y_true.astype(str))

    if unique_as_strings.issubset(set(string_labels)):
        has_string_values = y_true.dtype.kind in {"U", "S"} or (
            y_true.dtype.kind == "O" and all(isinstance(value, str) for value in y_true)
        )
        if has_string_values:
            return string_labels
        return numeric_labels

    return np.unique(y_true)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def proposed_pi_method_2(
        y_s1_gen,
        y_s2_gen,
        alpha=0.1, 
        eps=0.02, 
        D=1000, 
        max_m=None
    ):
    """
    Constructs a prediction interval for a single test point X_i following the refined 4 steps outlined.

    Args:
        y_s1_gen, y_s2_gen: see in ``get_y_gen_w_specified_size``. 
    """
    P_tilde = []
    candidate_intervals = []
    
    # We evaluate m up to max_m + 1 to check the (m+1) condition in Step 2
    for m in range(1, max_m + 2):
        # --- Step 1: Constructing Candidate Intervals ---
        # Generate D sets of size m + 1 using S1 (m for the mean, 1 for the independent synthetic sample)
        # print(f"m: {m}")
        samples_1 = get_y_gen_w_specified_size('1', m + 1, D, y_s1_gen, y_s2_gen) # shape: (D, m + 1)
        Z_m_1 = samples_1[:, :m]         # The m samples, shape (D, m)
        Z_star_1 = samples_1[:, m]       # The additional independent sample Z_tilde_1^{(*, d)}, shape (D,)
        
        # Compute the mean of the m samples
        means_1 = Z_m_1.mean(axis=1)     # Z_bar_1^{(m, d)}
        
        # Compute the synthetic residuals
        E_1 = Z_star_1 - means_1         # E_1^{(d)}
        
        # Define candidate residual interval I_m^(E) using alpha/4 and 1 - alpha/4 percentiles of E_1
        q_L1 = np.percentile(E_1, 100 * (alpha / 4))
        q_U1 = np.percentile(E_1, 100 * (1 - alpha / 4))
        candidate_intervals.append((q_L1, q_U1))
        
        # --- Step 2: Optimizing Synthetic Size through Tuning ---
        # Generate an independent synthetic sample using S2
        # print(f"m: {m}, y_s1_gen.shape: {y_s1_gen.shape}, y_s2_gen.shape: {y_s2_gen.shape}") # TEST
        samples_2 = get_y_gen_w_specified_size('2', m + 1, D, y_s1_gen, y_s2_gen)
        Z_star_2 = samples_2[:, m]       # Z_tilde_2^{(*, d)}
        
        # Compute cross-residual against the S1 mean estimator
        E_cross = Z_star_2 - means_1
        
        # P_tilde(I_m): Proportion of cross-residuals falling within the candidate interval I_m^(E)
        p_val = np.mean((E_cross >= q_L1) & (E_cross <= q_U1))
        P_tilde.append(p_val)

    # Find the optimal m_hat

    # Initialize trackers
    tracking_metrics = {
        "m_hat_fallback": False,
        "fallback_type": None,
        "disjoint": False,
        "disjoint_gap": 0.0
    }

    # Condition: P_tilde(I_m) >= 1 - alpha + eps AND P_tilde(I_{m+1}) < 1 - alpha + eps
    m_hat = None
    target_prob = 1 - alpha + eps
    for i in range(max_m):
        m = i + 1
        if P_tilde[i] >= target_prob and P_tilde[i+1] < target_prob:
            m_hat = m
            break
            
    # Fallback if theoretical condition isn't strictly met in the finite search space
    if m_hat is None:
        tracking_metrics["m_hat_fallback"] = True
        # Fallback: choose the smallest m that meets the primary threshold
        valid_ms = [i+1 for i, p in enumerate(P_tilde[:-1]) if p >= target_prob]
        if valid_ms:
            m_hat = valid_ms[0]
            tracking_metrics["fallback_type"] = "used_first_valid"
        else:
            m_hat = max_m
            tracking_metrics["fallback_type"] = "forced_max_m"

    # --- Step 3: Calculating the Interval Component (C1) ---
    # With the determined m_hat, repeat Step 1
    samples_1_hat = get_y_gen_w_specified_size('1', m_hat + 1, D, y_s1_gen, y_s2_gen)
    Z_m_1_hat = samples_1_hat[:, :m_hat]
    Z_star_1_hat = samples_1_hat[:, m_hat]
    
    means_1_hat = Z_m_1_hat.mean(axis=1)
    E_1_hat = Z_star_1_hat - means_1_hat
    
    q_L1_hat = np.percentile(E_1_hat, 100 * (alpha / 4))
    q_U1_hat = np.percentile(E_1_hat, 100 * (1 - alpha / 4))
    
    # Compute the total mean
    total_mean_1 = means_1_hat.mean()
    
    # Construct prediction interval C1
    C1_lower = total_mean_1 + q_L1_hat
    C1_upper = total_mean_1 + q_U1_hat
    
    # --- Step 4: Combining the Intervals (C2 and intersection) ---
    # Repeat Step 3 by replacing S1 with S2 to obtain C2
    samples_2_hat = get_y_gen_w_specified_size('2', m_hat + 1, D, y_s1_gen, y_s2_gen)
    Z_m_2_hat = samples_2_hat[:, :m_hat]
    Z_star_2_hat = samples_2_hat[:, m_hat]
    
    means_2_hat = Z_m_2_hat.mean(axis=1)
    E_2_hat = Z_star_2_hat - means_2_hat
    
    q_L2_hat = np.percentile(E_2_hat, 100 * (alpha / 4))
    q_U2_hat = np.percentile(E_2_hat, 100 * (1 - alpha / 4))
    
    # Compute the total mean for S2
    total_mean_2 = means_2_hat.mean()
    
    C2_lower = total_mean_2 + q_L2_hat
    C2_upper = total_mean_2 + q_U2_hat
    
    # Bonferroni intersection C_bar = C1 \cap C2
    C_bar_lower = max(C1_lower, C2_lower)
    C_bar_upper = min(C1_upper, C2_upper)
    
    # Graceful handling if the intersection is empty (disjoint intervals)
    if C_bar_lower > C_bar_upper:
        tracking_metrics["disjoint"] = True
        tracking_metrics["disjoint_gap"] = C_bar_lower - C_bar_upper # How badly did it miss?
        C_bar_lower, C_bar_upper = (C1_lower + C2_lower)/2, (C1_upper + C2_upper)/2
        
    return (C_bar_lower, C_bar_upper), m_hat, tracking_metrics


def get_y_gen_w_specified_size(
        data_split: Literal['1', '2'], 
        m, 
        D,
        y_s1_gen,
        y_s2_gen, 
    ):
    """
    Used in ``proposed_pi_method``.

    Args:
        y_s1_gen, y_s2_gen: shape (gen_size, 1). gen_size should be >= max_m * D where max_m is as 
                            defined in ``proposed_pi_method``.
    
    Returns:
        array of shape (D, m)
    """
    # Choose the appropriate generator
    y_gen = y_s1_gen if data_split == '1' else y_s2_gen
    
    gen_size = y_gen.shape[0]
    assert gen_size >= m*D, f"gen_size ({gen_size}) must be >= m*D ({m*D})"
    # assert gen_size % D == 0, "gen_size must be divisible by D"
    group_size = gen_size // D
    assert group_size >= m, f"Each group size ({group_size}) is less than m ({m})"

    # Vectorized extraction of the first `m` elements across all `D` groups
    y_reshaped = y_gen[:D * group_size].reshape((D, group_size))
    m_samples = y_reshaped[:, :m] 
    
    return m_samples


#-------------------baselines and evaluation, plotting functions-------------------

def split_conformal_prediction(X_train, Y_train, X_calib, Y_calib, X_test, alpha=0.1):
    # Train point predictor
    model = GradientBoostingRegressor(random_state=42)
    # ravel Y_train for sklearn
    model.fit(X_train, Y_train.ravel()) 
    
    # Calibrate
    calib_preds = model.predict(X_calib)
    # ravel Y_calib to prevent silent broadcasting bugs if Y is shape (n, 1)
    residuals = np.abs(Y_calib.ravel() - calib_preds) 
    
    # Calculate conformal quantile directly using order statistics
    n = len(residuals)
    
    # Calculate the exact k-th order statistic index
    # We want the ceiling of (n + 1) * (1 - alpha)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    
    # Cap k at n to prevent out-of-bounds if alpha is very small
    k = min(k, n)
    
    # Sort the residuals and extract the k-th smallest value
    Q = np.sort(residuals)[k - 1]

    # Predict intervals for test
    test_preds = model.predict(X_test)
    intervals = [(pred - Q, pred + Q) for pred in test_preds]
    
    return test_preds, intervals


def quantile_regression_baseline(X_train, Y_train, X_test, alpha=0.1):
    """ Baseline 2: Quantile Regression using Gradient Boosting """
    # Train Point/Median model for center of distributions
    median_model = GradientBoostingRegressor(loss='quantile', alpha=0.5, random_state=42)
    median_model.fit(X_train, Y_train.ravel())

    # Train Lower Bound model
    lower_model = GradientBoostingRegressor(loss='quantile', alpha=alpha/2, random_state=42)
    lower_model.fit(X_train, Y_train.ravel())
    
    # Train Upper Bound model
    upper_model = GradientBoostingRegressor(loss='quantile', alpha=1-alpha/2, random_state=42)
    upper_model.fit(X_train, Y_train.ravel())
    
    test_preds = median_model.predict(X_test)
    lower_preds = lower_model.predict(X_test)
    upper_preds = upper_model.predict(X_test)
    
    intervals = list(zip(lower_preds, upper_preds))
    return test_preds, intervals


def evaluate_intervals(intervals, Y_true, alpha=0.1):
    """ Calculates Coverage Probability, Avg Width, Std Width, and Winkler Score. """
    Y_true = Y_true.ravel()
    
    coverage = np.mean([lower <= y <= upper for (lower, upper), y in zip(intervals, Y_true)])
    widths = [upper - lower for lower, upper in intervals]
    
    # Winkler Score: Penalizes intervals that do not cover true y
    winkler_scores =[]
    for (lower, upper), y in zip(intervals, Y_true):
        width = upper - lower
        if y < lower:
            score = width + (2 / alpha) * (lower - y)
        elif y > upper:
            score = width + (2 / alpha) * (y - upper)
        else:
            score = width
        winkler_scores.append(score)
        
    return coverage, np.mean(widths), np.std(widths), np.mean(winkler_scores)


def plot_sorted_intervals(Y_true, intervals_dict, save_path):
    """ Visualizes prediction intervals against true values, sorted by true Y. """
    Y_true = Y_true.ravel()
    sorted_idx = np.argsort(Y_true)
    Y_true_sorted = Y_true[sorted_idx]
    
    valid_methods = {k: v for k, v in intervals_dict.items() if len(v) > 0}
    num_methods = len(valid_methods)
    
    if num_methods == 0: return

    fig, axes = plt.subplots(num_methods, 1, figsize=(10, 3.5 * num_methods), sharex=True)
    if num_methods == 1: axes = [axes]
    
    x = np.arange(len(Y_true_sorted))
    
    for ax, (name, intervals) in zip(axes, valid_methods.items()):
        intervals_sorted = [intervals[i] for i in sorted_idx]
        lower = np.array([iv[0] for iv in intervals_sorted])
        upper = np.array([iv[1] for iv in intervals_sorted])
        
        # Shade the prediction interval
        ax.fill_between(x, lower, upper, alpha=0.3, color='blue', label=f"{name} Interval")
        # Scatter the true Y values
        ax.scatter(x, Y_true_sorted, color='black', s=3, label="True Y")
        
        ax.set_title(f"{name} Intervals (Sorted by True Y)")
        ax.set_ylabel("Y value")
        ax.legend(loc="upper left")
        
    axes[-1].set_xlabel("Sample Index")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "1_sorted_intervals.png"), dpi=300)
    plt.close()


def plot_width_distributions(intervals_dict, save_path):
    """ Plots a boxplot comparing interval width distributions. """
    import seaborn as sns
    data, labels = [],[]
    for name, intervals in intervals_dict.items():
        if len(intervals) > 0:
            widths = [u - l for l, u in intervals]
            data.extend(widths)
            labels.extend([name] * len(widths))
            
    if not data: return
            
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=labels, y=data, palette="Set2")
    plt.title("Distribution of Interval Widths")
    plt.ylabel("Interval Width")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "2_width_distributions.png"), dpi=300)
    plt.close()


def plot_m_hat_distribution(m_hats, save_path):
    """ Plots what values of m_hat the proposed method chose. """
    import seaborn as sns
    if not m_hats: return
    
    plt.figure(figsize=(7, 4))
    sns.histplot(m_hats, bins=range(min(m_hats), max(m_hats) + 2), kde=False, color='coral')
    plt.title("Distribution of Chosen m_hat (Proposed Method)")
    plt.xlabel("m_hat size")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "3_m_hat_dist.png"), dpi=300)
    plt.close()
