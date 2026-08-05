"""
Functions for saving models, metrics, and hyperparameters.
"""
import os
import json
import pandas as pd
import torch


def save_model(model, outdir, epoch=None):
    os.makedirs(outdir, exist_ok=True)
    if epoch is None:
        weight_name = "tabtext_align.pt"
    else:
        epoch = int(epoch)
        weight_name = f"tabtext_align_epoch{epoch}.pt"
    path = os.path.join(outdir, weight_name)
    torch.save(model.state_dict(), path)


def save_metrics(metrics, outdir, epoch=None):
    # metrics: dict
    os.makedirs(outdir, exist_ok=True)
    fname = f"metrics.json" if epoch is None else f"metrics_epoch{epoch}.json"
    path = os.path.join(outdir, fname)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def append_metrics_to_csv(metrics, eval_dir, csv_name="metrics_log.csv"):
    os.makedirs(eval_dir, exist_ok=True)
    csv_path = os.path.join(eval_dir, csv_name)
    df_new = pd.DataFrame([metrics])
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(csv_path, index=False)


def save_hyperparams(args, save_dir):
    """Save hyperparameters (args namespace) to a JSON file in save_dir."""
    os.makedirs(save_dir, exist_ok=True)
    hyperparams_path = os.path.join(save_dir, "hyperparameters.json")
    # Convert argparse.Namespace to dict
    args_dict = vars(args)
    with open(hyperparams_path, "w") as f:
        json.dump(args_dict, f, indent=2)
    print(f"Saved hyperparameters to {hyperparams_path}")