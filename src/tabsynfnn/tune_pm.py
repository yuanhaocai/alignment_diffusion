# Portions adapted from amazon-science/TabSyn (Apache-2.0) and modified.
import os
import sys
import logging
from typing import Literal

import numpy as np
import optuna
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_squared_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.preprocessing import StandardScaler

from .prediction_interval_utils import (
    expected_calibration_error, multiclass_brier_score,
    infer_class_labels, labels_to_indices
)
from .tabsyn.main_pm import diffusion_train_pm
from .tabsyn.sample_pm import diffusion_repeated_sample_pm
from .utils_tuning import Record_best_trials, metric_dict_write
from .util_functions import quadratic_weighted_kappa


CLS_METRICS = ['acc', 'f1']
REG_METRICS = ['rmse', 'mse']

class Objective_diff:
    def __init__(
            self,
            use_val,
            y_emb_path,
            y_type,
            y_num_classes,
            use_tab,
            use_text,
            use_image,
            emb_alignment_path,
            vae_dat_path,
            text_path,
            image_path,
            text_pooling,
            normalize_tab_cond,
            normalize_text_tokens,
            normalize_text_pooled,
            normalize_projected_cond,
            d_in,
            es_thrshd,
            data_path, 
            tune_path, 
            sample_num_repeats,
            metric,
            dataname,
            study_name,
            wandb_log_interval,
            test_mode,
            seed=None,
            use_wandb=True,
        ):
        if not use_val:
            raise ValueError(
                "Hyperparameter selection requires validation data. Use a "
                "separate final-evaluation command for test data."
            )
        self.use_val = use_val
        self.y_emb_path = y_emb_path
        self.y_type = y_type
        self.y_num_classes = y_num_classes
        self.use_tab = use_tab
        self.use_text = use_text
        self.use_image = use_image
        self.emb_alignment_path = emb_alignment_path
        self.vae_dat_path = vae_dat_path
        self.text_path = text_path
        self.image_path = image_path
        self.text_pooling = text_pooling
        self.normalize_tab_cond = normalize_tab_cond
        self.normalize_text_tokens = normalize_text_tokens
        self.normalize_text_pooled = normalize_text_pooled
        self.normalize_projected_cond = normalize_projected_cond
        self.d_in = d_in
        self.es_thrshd = es_thrshd
        self.data_path = data_path
        self.tune_path = tune_path
        self.sample_num_repeats = sample_num_repeats
        self.metric = metric
        self.dataname = dataname
        self.study_name = study_name
        self.wandb_log_interval = wandb_log_interval
        self.test_mode = test_mode
        self.seed = seed
        self.use_wandb = use_wandb

    def __call__(self, trial):    
        # training
        if self.test_mode:
            epochs = trial.suggest_categorical('diffusion_epochs', [2])
            dim_t = trial.suggest_categorical('dim_t', [128])
        else:
            epochs = trial.suggest_categorical('diffusion_epochs', [5000]) # [1000] [100, 200]
            dim_t = trial.suggest_categorical('dim_t', [128, 256, 512, 768, 1024, 1536])
        batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048, 4096])
        lr = trial.suggest_float('lr', 5e-6, 1e-3, log=True)

        # sampling
        if self.test_mode:
            sampling_steps = trial.suggest_categorical('sample_steps', [6])
        else:
            sampling_steps = trial.suggest_categorical('sample_steps', [50, 100]) # [50]

        diffusion_path = os.path.join(self.tune_path, f'{trial.number}/diffusion')
        sampling_path = os.path.join(self.tune_path, f'{trial.number}/sampling')

        diffusion_train_pm(
            y_emb_path=self.y_emb_path,
            emb_alignment_path=self.emb_alignment_path,
            diffusion_path=diffusion_path,
            y_type=self.y_type,
            use_tab=self.use_tab,
            use_text=self.use_text,
            use_image=self.use_image,
            vae_dat_path=self.vae_dat_path,
            text_path=self.text_path,
            image_path=self.image_path,
            text_pooling=self.text_pooling,
            normalize_tab_cond=self.normalize_tab_cond,
            normalize_text_tokens=self.normalize_text_tokens,
            normalize_text_pooled=self.normalize_text_pooled,
            normalize_projected_cond=self.normalize_projected_cond,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            d_in=self.d_in,
            dim_t=dim_t,
            early_stopping_thrshd=self.es_thrshd,
            clip_grad=True,
            clip_grad_max_norm=2.0,
            use_wandb=(self.use_wandb and not self.test_mode),
            wandb_project=f'tabsyn_alignment_{self.dataname}',
            run_group=self.study_name,
            run_name=f'dim_t{dim_t}_lr{lr}_batch{batch_size}', # add more
            wandb_log_interval=self.wandb_log_interval,
            seed=None if self.seed is None else self.seed + trial.number,
        )

        diffusion_repeated_sample_pm(
            data_path=self.data_path,
            diffusion_path=diffusion_path,
            sampling_path=sampling_path,
            batch_size=1024,
            steps=sampling_steps,
            num_repeats=self.sample_num_repeats,
            use_val=self.use_val,
            base_seed=42,
        )

        metrics = {}
        y_train = np.load(os.path.join(self.data_path, 'y_train.npy'), allow_pickle=True)
        print("\nUsing validation set to compute metric in tuning.\n")
        y_eval = np.load(os.path.join(self.data_path, 'y_val.npy'), allow_pickle=True)
        if self.metric in CLS_METRICS:
            y_gen_eval = np.load(os.path.join(sampling_path, 'y_test_majority_vote.npy'), allow_pickle=True)
            y_gen_proba = np.load(
                os.path.join(sampling_path, "y_proba.npy"),
                allow_pickle=True,
            )
            class_labels = infer_class_labels(y_eval, self.y_num_classes)
            y_eval_indices = labels_to_indices(y_eval, class_labels)
            y_gen_indices = labels_to_indices(y_gen_eval, class_labels)
            metrics['acc'] = accuracy_score(y_eval_indices, y_gen_indices)
            metrics['qwk'] = quadratic_weighted_kappa(
                y_eval_indices, y_gen_indices, self.y_num_classes
            )
            if self.y_num_classes == 2:
                positive_proba = np.asarray(y_gen_proba, dtype=float).reshape(-1)
                metrics["ece"] = expected_calibration_error(
                    y_eval_indices, positive_proba
                )
                threshold_grid = np.linspace(0.0, 1.0, 101)
                threshold_scores = [
                    f1_score(
                        y_eval_indices,
                        positive_proba >= threshold,
                        zero_division=0,
                    )
                    for threshold in threshold_grid
                ]
                best_threshold_index = int(np.argmax(threshold_scores))
                metrics['f1'] = float(threshold_scores[best_threshold_index])
                metrics['validation_threshold'] = float(
                    threshold_grid[best_threshold_index]
                )
                metrics["roc_auc"] = roc_auc_score(
                    y_eval_indices,
                    positive_proba,
                )
                precision, recall, _ = precision_recall_curve(
                    y_eval_indices, positive_proba, pos_label=1
                )
                metrics['pr_auc'] = auc(recall, precision)
                metrics['log_loss'] = log_loss(
                    y_eval_indices, positive_proba, labels=[0, 1]
                )
                metrics["brier"] = brier_score_loss(
                    y_eval_indices,
                    positive_proba,
                )
            else:
                metrics["ece"] = expected_calibration_error(
                    y_eval_indices, y_gen_proba
                )
                metrics['f1_macro'] = f1_score(
                    y_eval_indices, y_gen_indices, average='macro'
                )
                metrics["brier"] = multiclass_brier_score(
                    y_eval,
                    y_gen_proba,
                    labels=class_labels,
                )
        elif self.metric in REG_METRICS:
            y_gen_eval = np.load(os.path.join(sampling_path, 'y_test_average.npy'), allow_pickle=True)
            scaler = StandardScaler()
            _ = scaler.fit_transform(y_train)
            y_eval_scaled = scaler.transform(y_eval)
            y_gen_eval_scaled = scaler.transform(y_gen_eval)
            metrics['mse'] = mean_squared_error(y_eval_scaled, y_gen_eval_scaled)
            metrics['rmse'] = root_mean_squared_error(y_eval_scaled, y_gen_eval_scaled)
            metrics['r2'] = r2_score(y_eval_scaled, y_gen_eval_scaled)
        else:
            raise NotImplementedError(self.metric)

        if trial.number == 0:
            with open(os.path.join(self.tune_path, 'metrics_of_trials.txt'), 'w') as file:
                file.write(f'Using {self.metric} as tuning metric\n\n')

        with open(os.path.join(sampling_path, 'metrics.txt'), 'w') as file:
            metric_dict_write(file, metrics, title=None)
        with open(os.path.join(self.tune_path, 'metrics_of_trials.txt'), 'a') as file:
            metric_dict_write(file, metrics, title=f'Trial {trial.number}:')
    
        return metrics[self.metric]


def hyperpar_tune_pm(
        use_val,
        y_emb_path,
        y_type,
        y_num_classes,
        use_tab,
        use_text,
        use_image,
        emb_alignment_path,
        vae_dat_path,
        text_path,
        image_path,
        d_in,
        early_stopping_thrshd,
        data_path, 
        tune_path,
        sample_num_repeats,
        dataname,
        wandb_log_interval,
        metric: Literal['acc', 'f1', 'rmse', 'mse'], 
        text_pooling="mean",
        normalize_tab_cond=True,
        normalize_text_tokens=False,
        normalize_text_pooled=True,
        normalize_projected_cond=False,
        n_trials=50, 
        direction=None, 
        directions=None, 
        rdb=None,
        study_name=None, 
        test_mode=False
    ):
    """
    Tune hyperparameters for predictive modeling diffusion. Built upon ``hyperpar_tune_multimodal``.

    Args:
        tune_path: the path to save the weights. Then a subdirectories of f'{trial.number}/vae' will be 
                    created as the tuning proceeds.
        direction: the direction that `optuna` to optimize the objective function. Depends on the metric used.
        rdb: whether to save the study with RDB backend.
        study_name: the name for the file thats save tuning logs will be named `f'{study_name}-study'`, 
                    when `rdb` is `True`.
    """
    if not use_val:
        raise ValueError(
            "Hyperparameter tuning requires use_val=True."
        )

    if test_mode:
        n_trials = 3
        sample_num_repeats = 3

    assert metric in CLS_METRICS + REG_METRICS
    if metric in CLS_METRICS:
        assert isinstance(y_num_classes, int) and y_num_classes >= 2
    if metric == 'f1':
        assert y_num_classes == 2
    
    os.makedirs(tune_path, exist_ok=True)

    if rdb:
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        print(f"study_name: {study_name}")
        storage_nm = os.path.join(tune_path, f'{study_name}.db')
        storage_name = f'sqlite:///{storage_nm}'
        study = optuna.create_study(
            study_name=study_name, 
            storage=storage_name,
            sampler=optuna.samplers.TPESampler(seed=0), 
            direction=direction, 
            directions=directions,
            load_if_exists=True
        )
    else:
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=0), 
            direction=direction, 
            directions=directions
        )

    objective = Objective_diff(
        use_val,
        y_emb_path,
        y_type,
        y_num_classes,
        use_tab,
        use_text,
        use_image,
        emb_alignment_path,
        vae_dat_path,
        text_path,
        image_path,
        text_pooling,
        normalize_tab_cond,
        normalize_text_tokens,
        normalize_text_pooled,
        normalize_projected_cond,
        d_in,
        early_stopping_thrshd,
        data_path, 
        tune_path, 
        sample_num_repeats,
        metric,
        dataname,
        study_name,
        wandb_log_interval,
        test_mode
    )        

    record_best_trial = Record_best_trials(tune_path=tune_path)

    study.optimize(
        objective, 
        n_trials=n_trials, 
        show_progress_bar=True, 
        callbacks=[record_best_trial]
    )
