"""
Functions for evaluation.
"""
from typing import Optional, Literal
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from tabsynfnn.util_functions import quadratic_weighted_kappa

from .model import compute_combined_cosine_sim, unwrap_model


def to_number(val):
    if torch.is_tensor(val):
        return val.item()
    return float(val)


def evaluate(
        model, 
        dataloader, 
        y_type: Literal['continuous', 'categorical'], 
        y_num_classes: Optional[int] = None
    ):
    model.eval()
    retrieval_metrics = []
    preds = []
    preds_prob = [] 
    gts = []
    cls_total = 0

    if y_type == "categorical":
        assert isinstance(y_num_classes, int) and y_num_classes >= 2
        topk = (1,3) if y_num_classes >= 3 else (1,)
        topk_accumulators = {f"cls_top{k}_count": 0 for k in topk}

    device = next(unwrap_model(model).parameters()).device

    with torch.no_grad():
        for batch in dataloader:
            tab_emb, y, text_emb = batch
            tab_emb = tab_emb.float().to(device)
            text_emb = text_emb.float().to(device)
            y = y.to(device)
            
            # === retrieval/alignment metrics ===
            sim_orig = compute_combined_cosine_sim(tab_emb, text_emb)
            labels = torch.arange(sim_orig.shape[0], device=sim_orig.device)
            pred_orig = sim_orig.argmax(dim=1)
            acc_orig = (pred_orig == labels).float().mean().item()
            diag_mean_orig = sim_orig.diag().mean().item()
            non_diag_mean_orig = (sim_orig.sum() - sim_orig.diag().sum()) / (sim_orig.numel() - sim_orig.shape[0])

            tab_proj, text_proj, logit_scale = model(tab_emb, text_emb)
            sim_aligned = compute_combined_cosine_sim(tab_proj, text_proj)
            sim_aligned_scaled = sim_aligned * logit_scale.exp()

            pred_aligned = sim_aligned.argmax(dim=1)
            acc_aligned = (pred_aligned == labels).float().mean().item()
            diag_mean_aligned = sim_aligned.diag().mean().item()
            non_diag_mean_aligned = (sim_aligned.sum() - sim_aligned.diag().sum()) / (sim_aligned.numel() - sim_aligned.shape[0])

            retrieval_metrics.append({
                "diag_mean_orig": diag_mean_orig,
                "non_diag_mean_orig": non_diag_mean_orig,
                "retrieval_acc_orig": acc_orig,
                "diag_mean_aligned": diag_mean_aligned,
                "non_diag_mean_aligned": non_diag_mean_aligned,
                "retrieval_acc_aligned": acc_aligned,
                "diag_mean_aligned_with_temp": sim_aligned_scaled.diag().mean().item(),
            })

            if y_type == "categorical":
                logits = unwrap_model(model).head(tab_proj)        # [B, num_classes]
                pred = logits.argmax(dim=1)                            # Top-1

                if y_num_classes == 2:
                    probs = torch.softmax(logits, dim=1)
                    # Take probability of class 1 (assume positive is index 1; adjust if needed)
                    binary_probs = probs[:, 1].detach().cpu().numpy()
                    preds_prob.append(binary_probs)                # Collect these in a list                

                for k in topk:
                    topk_pred = logits.topk(k, dim=1).indices            # [B, k]
                    # Count how many GTs are in any of the top-k for the batch
                    topk_correct = (topk_pred == y.unsqueeze(-1)).any(dim=1).sum().item()
                    topk_accumulators[f"cls_top{k}_count"] += topk_correct
                cls_total += y.size(0)
            elif y_type == "continuous":
                pred = unwrap_model(model).head(tab_proj)
                pred = pred.squeeze(-1)   # [B]

            preds.append(pred.detach().cpu().numpy())
            gts.append(y.cpu().numpy())

    # Aggregate
    retrieval_keys = retrieval_metrics[0].keys()
    out_metrics = {k: float(np.mean([to_number(x[k]) for x in retrieval_metrics])) for k in retrieval_keys}
    
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(gts)

    if y_type == "categorical":
        out_metrics["cls_accuracy"] = accuracy_score(y_true, y_pred)
        if y_num_classes == 2:
            y_preds_prob = np.concatenate(preds_prob)
            out_metrics["cls_f1"] = f1_score(y_true, y_pred, average='binary')
            out_metrics["auc_roc"] = roc_auc_score(y_true, y_preds_prob)
        else:
            out_metrics["cls_macroF1"] = f1_score(y_true, y_pred, average='macro')
        for k in topk:
            out_metrics[f"cls_top{k}_acc"] = topk_accumulators[f"cls_top{k}_count"] / cls_total
        out_metrics['conf_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        out_metrics['class_report'] = classification_report(y_true, y_pred, output_dict=True)
        out_metrics['quadratic_weighted_kappa'] = quadratic_weighted_kappa(y_true, y_pred, num_ratings=y_num_classes)
    elif y_type == "continuous":
        out_metrics["mse"] = float(mean_squared_error(y_true, y_pred))
        out_metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        out_metrics["r2"] = float(r2_score(y_true, y_pred))

    return out_metrics
